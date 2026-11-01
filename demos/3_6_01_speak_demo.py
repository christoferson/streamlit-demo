import streamlit as st
import boto3
import cmn_settings
import cmn_constants
import cmn_security
import json
import logging
from io import BytesIO

from PIL import Image
import io
import base64
import pandas as pd
from cmn.bedrock_models import FOUNDATION_MODELS, FoundationModel
from datetime import datetime

from botocore.exceptions import ClientError, ReadTimeoutError

import streamlit as st
import boto3
import cmn_settings
import cmn_constants
import cmn_security
import json
import logging
from io import BytesIO
import asyncio
import base64
import uuid
from PIL import Image
import io
import pandas as pd
from cmn.bedrock_models import FOUNDATION_MODELS, FoundationModel
from datetime import datetime
import sounddevice as sd
import numpy as np
import wave
import threading
import queue
import pyaudio
from botocore.config import Config
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest
from botocore.credentials import Credentials
from botocore.exceptions import ClientError, ReadTimeoutError
from aws_crt.auth import AwsCredentialsProvider
from aws_crt.http import HttpAuthSchemeResolver, SigV4AuthScheme, HTTPAuthScheme
from aws_crt.io import ClientBootstrap, DefaultHostResolver, EventLoopGroup
from awscrt.io import ClientBootstrap, DefaultHostResolver, EventLoopGroup
from awscrt.auth import AwsCredentialsProvider
from awscrt.http import HttpAuthSchemeResolver, SigV4AuthScheme, HTTPAuthScheme
from awscrt.io import ClientBootstrap, DefaultHostResolver, EventLoopGroup

# Import the experimental bidirectional streaming SDK for Nova Sonic
from bedrock_runtime.bedrock_runtime_client import BedrockRuntimeClient
from bedrock_runtime.models import (
    InvokeModelWithBidirectionalStreamInput,
    InvokeModelWithBidirectionalStreamInputChunk,
    BidirectionalInputPayloadPart
)

AWS_REGION = cmn_settings.AWS_REGION
MAX_MESSAGES = 80 * 2

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Environment credentials resolver for bidirectional streaming
class EnvironmentCredentialsResolver:
    def __call__(self):
        session = boto3.Session()
        credentials = session.get_credentials()
        return AwsCredentialsProvider.new_default()

# Audio handling classes for recording and playback
class AudioRecorder:
    def __init__(self, sample_rate=16000, channels=1):
        self.sample_rate = sample_rate
        self.channels = channels
        self.audio_queue = queue.Queue()
        self.is_recording = False
        self.audio_data = []
        self.stream = None
        self.p = pyaudio.PyAudio()

    def callback(self, in_data, frame_count, time_info, status):
        if self.is_recording:
            self.audio_queue.put(in_data)
            self.audio_data.append(in_data)
        return (in_data, pyaudio.paContinue)

    def start_recording(self):
        self.is_recording = True
        self.audio_data = []
        self.stream = self.p.open(
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=1024,
            stream_callback=self.callback
        )
        self.stream.start_stream()

    def stop_recording(self):
        if self.stream and self.stream.is_active():
            self.stream.stop_stream()
            self.stream.close()
        self.is_recording = False
        return b''.join(self.audio_data)

    def get_audio_chunk(self, timeout=0.1):
        try:
            return self.audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def __del__(self):
        if self.stream:
            self.stream.close()
        self.p.terminate()

class AudioPlayer:
    def __init__(self, sample_rate=16000, channels=1):
        self.sample_rate = sample_rate
        self.channels = channels
        self.p = pyaudio.PyAudio()
        self.stream = None

    def play_audio(self, audio_bytes):
        if self.stream and self.stream.is_active():
            self.stream.stop_stream()
            self.stream.close()

        self.stream = self.p.open(
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=self.sample_rate,
            output=True
        )
        self.stream.write(audio_bytes)
        self.stream.stop_stream()
        self.stream.close()

    def __del__(self):
        if self.stream:
            self.stream.close()
        self.p.terminate()

class NovaSonicClient:
    def __init__(self, model_id="amazon.nova-sonic-v1:0", region=AWS_REGION):
        self.model_id = model_id
        self.region = region
        self.bedrock_client = None
        self.stream_response = None
        self.is_active = False
        self.audio_output_queue = asyncio.Queue()
        self.output_queue = asyncio.Queue()
        self.role = None
        self.display_assistant_text = False
        self.barge_in = False
        self.toolUseContent = None
        self.toolName = None
        self.toolUseId = None
        self._initialize_client()

    def _initialize_client(self):
        """Initialize the Bedrock client."""
        config = Config(
            endpoint_uri=f"https://bedrock-runtime.{self.region}.amazonaws.com",
            region=self.region,
            aws_credentials_identity_resolver=EnvironmentCredentialsResolver(),
            http_auth_scheme_resolver=HttpAuthSchemeResolver(),
            http_auth_schemes={"aws.auth#sigv4": SigV4AuthScheme()}
        )
        self.bedrock_client = BedrockRuntimeClient(config=config)

    async def start_session(self, temperature=0.7, top_p=0.9, max_tokens=1024):
        """Start a bidirectional streaming session with Nova Sonic."""
        self.stream_response = await self.bedrock_client.invoke_model_with_bidirectional_stream(
            InvokeModelWithBidirectionalStreamInput(model_id=self.model_id)
        )
        self.is_active = True

        # Start the response processing task
        asyncio.create_task(self._process_responses())

        # Send the session start event
        START_SESSION_EVENT = f'''{{
            "event": {{
                "sessionStart": {{
                "inferenceConfiguration": {{
                    "maxTokens": {max_tokens},
                    "topP": {top_p},
                    "temperature": {temperature}
                    }}
                }}
            }}
        }}'''

        event = InvokeModelWithBidirectionalStreamInputChunk(
            value=BidirectionalInputPayloadPart(bytes_=START_SESSION_EVENT.encode('utf-8'))
        )  
        try:
            await self.stream_response.input_stream.send(event)
        except Exception as e:
            print(f"Error sending session start event: {str(e)}")

    async def send_audio(self, audio_bytes):
        """Send audio data to the model."""
        if not self.is_active:
            print("Session not active")
            return

        # Convert audio bytes to base64
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')

        # Create the audio input event
        AUDIO_INPUT_EVENT = f'''{{
            "event": {{
                "audioInput": {{
                    "content": "{audio_base64}"
                }}
            }}
        }}'''

        event = InvokeModelWithBidirectionalStreamInputChunk(
            value=BidirectionalInputPayloadPart(bytes_=AUDIO_INPUT_EVENT.encode('utf-8'))
        )

        try:
            await self.stream_response.input_stream.send(event)
        except Exception as e:
            print(f"Error sending audio: {str(e)}")

    async def send_text(self, text):
        """Send text to the model."""
        if not self.is_active:
            print("Session not active")
            return

        # Create the text input event
        TEXT_INPUT_EVENT = f'''{{
            "event": {{
                "textInput": {{
                    "content": "{text}"
                }}
            }}
        }}'''

        event = InvokeModelWithBidirectionalStreamInputChunk(
            value=BidirectionalInputPayloadPart(bytes_=TEXT_INPUT_EVENT.encode('utf-8'))
        )

        try:
            await self.stream_response.input_stream.send(event)
        except Exception as e:
            print(f"Error sending text: {str(e)}")

    async def send_tool_start_event(self, tool_content_id):
        """Send tool start event."""
        TOOL_START_EVENT = f'''{{
            "event": {{
                "toolStart": {{
                    "toolContentId": "{tool_content_id}"
                }}
            }}
        }}'''

        event = InvokeModelWithBidirectionalStreamInputChunk(
            value=BidirectionalInputPayloadPart(bytes_=TOOL_START_EVENT.encode('utf-8'))
        )

        try:
            await self.stream_response.input_stream.send(event)
        except Exception as e:
            print(f"Error sending tool start event: {str(e)}")

    async def send_tool_result_event(self, tool_content_id, tool_result):
        """Send tool result event."""
        TOOL_RESULT_EVENT = f'''{{
            "event": {{
                "toolResult": {{
                    "toolContentId": "{tool_content_id}",
                    "content": "{tool_result}"
                }}
            }}
        }}'''

        event = InvokeModelWithBidirectionalStreamInputChunk(
            value=BidirectionalInputPayloadPart(bytes_=TOOL_RESULT_EVENT.encode('utf-8'))
        )

        try:
            await self.stream_response.input_stream.send(event)
        except Exception as e:
            print(f"Error sending tool result event: {str(e)}")

    async def send_tool_content_end_event(self, tool_content_id):
        """Send tool content end event."""
        TOOL_CONTENT_END_EVENT = f'''{{
            "event": {{
                "contentEnd": {{
                    "toolContentId": "{tool_content_id}"
                }}
            }}
        }}'''

        event = InvokeModelWithBidirectionalStreamInputChunk(
            value=BidirectionalInputPayloadPart(bytes_=TOOL_CONTENT_END_EVENT.encode('utf-8'))
        )

        try:
            await self.stream_response.input_stream.send(event)
        except Exception as e:
            print(f"Error sending tool content end event: {str(e)}")

    async def _process_responses(self):
        """Process incoming responses from Bedrock."""
        try:            
            while self.is_active:
                try:
                    output = await self.stream_response.await_output()
                    result = await output[1].receive()
                    if result.value and result.value.bytes_:
                        try:
                            response_data = result.value.bytes_.decode('utf-8')
                            json_data = json.loads(response_data)

                            # Handle different response types
                            if 'event' in json_data:
                                if 'contentStart' in json_data['event']:
                                    content_start = json_data['event']['contentStart']
                                    # set role
                                    self.role = content_start['role']
                                    # Check for speculative content
                                    if 'additionalModelFields' in content_start:
                                        try:
                                            additional_fields = json.loads(content_start['additionalModelFields'])
                                            if additional_fields.get('generationStage') == 'SPECULATIVE':
                                                self.display_assistant_text = True
                                            else:
                                                self.display_assistant_text = False
                                        except json.JSONDecodeError:
                                            print("Error parsing additionalModelFields")
                                elif 'textOutput' in json_data['event']:
                                    text_content = json_data['event']['textOutput']['content']
                                    role = json_data['event']['textOutput']['role']
                                    # Check if there is a barge-in
                                    if '{ "interrupted" : true }' in text_content:
                                        self.barge_in = True

                                    if (self.role == "ASSISTANT" and self.display_assistant_text):
                                        print(f"Assistant: {text_content}")
                                    elif (self.role == "USER"):
                                        print(f"User: {text_content}")

                                elif 'audioOutput' in json_data['event']:
                                    audio_content = json_data['event']['audioOutput']['content']
                                    audio_bytes = base64.b64decode(audio_content)
                                    await self.audio_output_queue.put(audio_bytes)
                                elif 'toolUse' in json_data['event']:
                                    self.toolUseContent = json_data['event']['toolUse']
                                    self.toolName = json_data['event']['toolUse']['toolName']
                                    self.toolUseId = json_data['event']['toolUse']['toolUseId']
                                elif 'contentEnd' in json_data['event'] and json_data['event'].get('contentEnd', {}).get('type') == 'TOOL':
                                    toolResult = await self.processToolUse(self.toolName, self.toolUseContent)
                                    toolContent = str(uuid.uuid4())
                                    await self.send_tool_start_event(toolContent)
                                    await self.send_tool_result_event(toolContent, toolResult)
                                    await self.send_tool_content_end_event(toolContent)
                                elif 'completionEnd' in json_data['event']:
                                    # Handle end of conversation, no more response will be generated
                                    print("End of response sequence")

                            # Put the response in the output queue for other components
                            await self.output_queue.put(json_data)
                        except json.JSONDecodeError:
                            await self.output_queue.put({"raw_data": response_data})
                except StopAsyncIteration:
                    # Stream has ended
                    break
                except Exception as e:
                   # Handle ValidationException properly
                    if "ValidationException" in str(e):
                        error_message = str(e)
                        print(f"Validation error: {error_message}")
                    else:
                        print(f"Error receiving response: {e}")
                    break

        except Exception as e:
            print(f"Response processing error: {e}")
        finally:
            self.is_active = False

    async def processToolUse(self, toolName, toolUseContent):
        """Process tool use requests."""
        # Implement tool processing logic here
        return "Tool result placeholder"

    async def close(self):
        """Close the streaming session."""
        self.is_active = False
        if self.stream_response:
            try:
                await self.stream_response.input_stream.end()
            except Exception as e:
                print(f"Error closing stream: {str(e)}")

    async def get_audio_output(self, timeout=None):
        """Get audio output from the queue."""
        try:
            return await asyncio.wait_for(self.audio_output_queue.get(), timeout)
        except asyncio.TimeoutError:
            return None


####

# Add these to your existing Streamlit UI setup
def initialize_session_state():
    """Initialize session state variables for speech-to-speech functionality"""
    if "nova_sonic_session_active" not in st.session_state:
        st.session_state.nova_sonic_session_active = False
    if "audio_recorder" not in st.session_state:
        st.session_state.audio_recorder = AudioRecorder()
    if "audio_player" not in st.session_state:
        st.session_state.audio_player = AudioPlayer()
    if "recording" not in st.session_state:
        st.session_state.recording = False
    if "audio_output_buffer" not in st.session_state:
        st.session_state.audio_output_buffer = []
    if "conversation_text" not in st.session_state:
        st.session_state.conversation_text = []
    if "nova_client" not in st.session_state:
        st.session_state.nova_client = None
    if "async_tasks" not in st.session_state:
        st.session_state.async_tasks = []

def start_recording():
    """Start audio recording"""
    if not st.session_state.recording:
        st.session_state.audio_recorder.start_recording()
        st.session_state.recording = True

def stop_recording():
    """Stop audio recording and process the audio"""
    if st.session_state.recording:
        audio_data = st.session_state.audio_recorder.stop_recording()
        st.session_state.recording = False

        # Process the recorded audio
        if audio_data and st.session_state.nova_sonic_session_active:
            # Create a new task to send the audio to Nova Sonic
            asyncio.create_task(send_audio_to_nova(audio_data))

            # Add user audio to conversation history
            st.session_state.conversation_text.append({"role": "user", "content": "[Audio Input]"})

async def send_audio_to_nova(audio_data):
    """Send audio data to Nova Sonic"""
    if st.session_state.nova_client and st.session_state.nova_sonic_session_active:
        await st.session_state.nova_client.send_audio(audio_data)

async def start_nova_sonic_session():
    """Start a new Nova Sonic session"""
    if st.session_state.nova_sonic_session_active:
        return

    # Create a new Nova Sonic client
    st.session_state.nova_client = NovaSonicClient(
        model_id="amazon.nova-sonic-v1:0",
        region=AWS_REGION
    )

    # Start the session with the configured parameters
    await st.session_state.nova_client.start_session(
        temperature=st.session_state.temperature,
        top_p=st.session_state.top_p,
        max_tokens=st.session_state.max_tokens
    )

    st.session_state.nova_sonic_session_active = True

    # Start the audio output processing task
    asyncio.create_task(process_audio_output())

async def end_nova_sonic_session():
    """End the Nova Sonic session"""
    if st.session_state.nova_client:
        await st.session_state.nova_client.close()
    st.session_state.nova_sonic_session_active = False

async def process_audio_output():
    """Process audio output from Nova Sonic"""
    while st.session_state.nova_sonic_session_active:
        try:
            # Get audio output from the Nova Sonic client
            audio_bytes = await st.session_state.nova_client.get_audio_output(timeout=0.5)
            if audio_bytes:
                # Play the audio
                st.session_state.audio_player.play_audio(audio_bytes)

                # Store the audio for potential replay
                st.session_state.audio_output_buffer.append(audio_bytes)
        except Exception as e:
            print(f"Error processing audio output: {e}")
            await asyncio.sleep(0.1)
    
# Add this to your Streamlit UI section
def create_speech_to_speech_ui():
    """Create the speech-to-speech UI components"""
    st.markdown("## 🎙️ Speech-to-Speech Conversation")

    # Initialize session state
    initialize_session_state()

    # Session control buttons
    col1, col2 = st.columns(2)
    with col1:
        if not st.session_state.nova_sonic_session_active:
            if st.button("Start Speech Session", type="primary"):
                # Use asyncio to start the Nova Sonic session
                asyncio.run(start_nova_sonic_session())
                st.rerun()
        else:
            st.success("Session Active")

    with col2:
        if st.session_state.nova_sonic_session_active:
            if st.button("End Speech Session", type="secondary"):
                # Use asyncio to end the Nova Sonic session
                asyncio.run(end_nova_sonic_session())
                st.rerun()

    # Recording controls
    if st.session_state.nova_sonic_session_active:
        st.markdown("### Recording Controls")
        col1, col2 = st.columns(2)

        with col1:
            if not st.session_state.recording:
                if st.button("Start Recording 🎤", type="primary"):
                    start_recording()
                    st.rerun()
            else:
                st.warning("Recording in progress...")

        with col2:
            if st.session_state.recording:
                if st.button("Stop Recording ⏹️", type="secondary"):
                    # Use a thread to stop recording to avoid blocking
                    threading.Thread(target=stop_recording).start()
                    st.rerun()

    # Display conversation history
    st.markdown("### Conversation History")
    conversation_container = st.container()

    with conversation_container:
        for message in st.session_state.conversation_text:
            with st.chat_message(message["role"]):
                st.write(message["content"])

    # Text input as an alternative to speech
    if st.session_state.nova_sonic_session_active:
        text_input = st.chat_input("Type a message (alternative to speech)")
        if text_input:
            # Add user text to conversation history
            st.session_state.conversation_text.append({"role": "user", "content": text_input})

            # Send text to Nova Sonic
            if st.session_state.nova_client:
                asyncio.run(st.session_state.nova_client.send_text(text_input))

            st.rerun()



# https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/bedrock-runtime/client/converse_stream.html

# include up to 20 images. Each image's size, height, and width must be no more than 3.75 MB, 8,000 px, and 8,000 px, respectively.
#  include up to five documents. Each document's size must be no more than 5 MB.
# can only include images and documents if the role is user.

###### AUTH START #####

#if not cmn_auth.check_password():
#   st.stop()

######  AUTH END #####

####################################################################################

bedrock_runtime = boto3.client('bedrock-runtime', region_name=AWS_REGION)
bedrock_runtime_us_west_2 = boto3.client('bedrock-runtime', region_name="us-west-2")
cloudwatch_logs = boto3.client('logs', region_name=AWS_REGION)
polly = boto3.client("polly", region_name=AWS_REGION)

####################################################################################

st.set_page_config(
    page_title="Converse",
    page_icon="🧊",
    layout="centered", # "centered" or "wide"
    initial_sidebar_state="expanded", #"auto", "expanded", or "collapsed"
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': None
    },
)


st.logo(icon_image="images/logo.png", image="images/logo_text.png")

st.markdown(cmn_constants.css_button_primary, unsafe_allow_html=True)

def image_to_base64(image,mime_type:str):
    buffer = io.BytesIO()
    image.save(buffer, format=mime_type)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def trim_conversation_history(messages):
    """Ensure conversation doesn't exceed maximum length"""
    messages_len = len(messages)
    if messages_len > MAX_MESSAGES:
        # Remove oldest messages while keeping pairs together
        return messages[-(MAX_MESSAGES * 2):]
    return messages

def validate_conversation_structure(conversation):
    """Validate the structure of uploaded conversation data"""
    if not isinstance(conversation, list):
        raise ValueError("Invalid format: expected a list of messages")

    if len(conversation) > MAX_MESSAGES:
        raise ValueError(f"Conversation exceeds maximum length of {MAX_MESSAGES} messages")

    for message in conversation:
        if not isinstance(message, dict):
            raise ValueError("Invalid message format: expected a dictionary")

        if "role" not in message or "content" not in message:
            raise ValueError("Message missing required fields: 'role' and 'content'")

        if message["role"] not in ["user", "assistant", "system"]:
            raise ValueError(f"Invalid role: {message['role']}")

        if not isinstance(message["content"], list):
            raise ValueError("Invalid content format: expected a list")

        for content_item in message["content"]:
            if not isinstance(content_item, dict) or "text" not in content_item:
                raise ValueError("Invalid content item: expected 'text' key")

def get_conversation_download():
    """Download conversation with security signature"""
    if not st.session_state.menu_speak_messages:
        raise ValueError("No conversation to download")

    security = cmn_security.ConversationSecurity()
    secure_package = security.secure_conversation(
        trim_conversation_history(st.session_state.menu_speak_messages)
    )
    return BytesIO(json.dumps(secure_package, indent=2).encode())

def clear_conversation():
    """Clear conversation state"""
    st.session_state.menu_speak_messages = []
    st.session_state.menu_speak_uploader_key = 0

def upload_conversation(uploaded_file):
    """Upload and verify conversation integrity"""
    try:
        security = cmn_security.ConversationSecurity()

        # Read and parse the uploaded file
        content = uploaded_file.getvalue().decode("utf-8")
        conversation_package = json.loads(content)

        # Verify integrity
        if not security.verify_conversation(conversation_package):
            st.error("Security verification failed. The conversation may have been tampered with.")
            return False

        # Validate structure
        validate_conversation_structure(conversation_package["messages"])

        # Update session state
        st.session_state.menu_speak_messages = conversation_package["messages"]
        st.session_state.menu_speak_uploader_key = 0

        st.success("Conversation uploaded and verified successfully!")
        return True

    except json.JSONDecodeError:
        st.error("Invalid JSON file format")
    except ValueError as e:
        st.error(f"Error in conversation structure: {str(e)}")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        logging.error(f"Upload error: {str(e)}", exc_info=True)
    return False

# Get the top 10 users with the most tokens in a given period (e.g., past 3 months):
# fields @timestamp, @message
# | parse @message '{"user_name":"*","input_tokens":*,"output_tokens":*,"total_tokens":*,"event_time":"*"}' as user_name, input_tokens, output_tokens, total_tokens, event_time
# | filter ispresent(total_tokens) and event_time > '2023-03-01T00:00:00' and event_time < '2023-06-01T00:00:00'
# | stats sum(total_tokens) as total_tokens_used by user_name
# | sort total_tokens_used desc
# | limit 10
#Get the total tokens used by a particular user for the whole period:
# fields @timestamp, @message
# | parse @message '{"user_name":"*","input_tokens":*,"output_tokens":*,"total_tokens":*,"event_time":"*"}' as user_name, input_tokens, output_tokens, total_tokens, event_time
# | filter user_name = 'specific_user_name' and ispresent(total_tokens)
# | stats sum(total_tokens) as total_tokens_used by user_name
def push_to_cloudwatch(user_name, input_tokens, output_tokens, total_tokens):
    log_group_name = "/app/openai/metrics/invocations"
    log_stream_name = datetime.now().strftime("%Y-%m-%d")

    # Ensure the log group exists
    try:
        cloudwatch_logs.create_log_group(logGroupName=log_group_name)
    except cloudwatch_logs.exceptions.ResourceAlreadyExistsException as e:
        #logger.error(f"Failed to create log group to CloudWatch: {str(e)}")
        pass

    # Ensure the log stream exists
    try:
        cloudwatch_logs.create_log_stream(logGroupName=log_group_name, logStreamName=log_stream_name)
    except cloudwatch_logs.exceptions.ResourceAlreadyExistsException as e:
        #logger.error(f"Failed to create log stream to CloudWatch: {str(e)}")
        pass

    # Create the log entry as a JSON object
    log_entry = {
        'timestamp': int(datetime.now().timestamp() * 1000),
        'message': json.dumps({
            'user_name': user_name,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'total_tokens': total_tokens,
            'event_time': datetime.now().isoformat()
        })
    }

    # Push the log entry to CloudWatch
    try:
        cloudwatch_logs.put_log_events(
            logGroupName=log_group_name,
            logStreamName=log_stream_name,
            logEvents=[log_entry]
        )
    except Exception as e:
        logger.error(f"Failed to push logs to CloudWatch: {str(e)}")


def delete_message_pair(index):
    if index < len(st.session_state.menu_speak_messages):
        del st.session_state.menu_speak_messages[index:index+2]
    if f"feedback_{index+1}" in st.session_state.menu_speak_messages_feedback:
        del st.session_state.menu_speak_messages_feedback[f"feedback_{index+1}"]

mime_mapping = {
    "image/png": "png",
    "image/jpeg": "jpeg",
    "image/jpg": "jpeg",
    "image/gif": "gif",
    "image/webp": "webp",
}

mime_mapping_image = {
    "image/png": "png",
    "image/jpeg": "jpeg",
    "image/jpg": "jpeg",
    "image/gif": "gif",
    "image/webp": "webp",
}

#'pdf'|'csv'|'doc'|'docx'|'xls'|'xlsx'|'html'|'txt'|'md',
mime_mapping_document = {
    "text/plain": "txt",
    "application/vnd.ms-excel": "csv",
    "application/pdf": "pdf",
}
       
# https://docs.aws.amazon.com/bedrock/latest/userguide/model-ids.html
opt_model_id_list = [
    "amazon.nova-sonic-v1:0",
]

# "You are a question and answering chatbot"
# - Maintaining a helpful and courteous tone throughout the interaction.
opt_system_msg_int = """You are a friend. The user and you will engage in a spoken dialog exchanging the transcripts of a natural real-time conversation. Keep your responses short, generally two or three sentences for chatty scenarios.
"""

if "menu_speak_messages" not in st.session_state:
    st.session_state["menu_speak_messages"] = []

# Add this to store feedback
if "menu_speak_messages_feedback" not in st.session_state:
    st.session_state["menu_speak_messages_feedback"] = {}
    
with st.sidebar:
    opt_model_id = st.selectbox(label="Model ID", options=opt_model_id_list, index = 0, key="model_id")

opt_fm:FoundationModel = FOUNDATION_MODELS.find(opt_model_id)

opt_fm_max_tokens = opt_fm.inference_parameters.max_tokens
opt_fm_top_k = opt_fm.inference_parameters.top_k

with st.sidebar:
    opt_temperature = st.slider(label="Temperature", min_value=0.0, max_value=1.0, value=0.1, step=0.1, key="temperature")
    opt_top_p = st.slider(label="Top P", min_value=0.0, max_value=1.0, value=1.0, step=0.1, key="top_p")
    if opt_fm_top_k.is_supported:
        opt_top_k = st.slider(label="Top K", min_value=0, max_value=500, value=250, step=1, key="top_k")
    else:
        opt_top_k = 0
    opt_max_tokens = st.slider(label="Max Tokens", min_value=opt_fm_max_tokens.min_value, max_value=opt_fm_max_tokens.max_value, value=opt_fm_max_tokens.default_value, step=1, key="max_tokens")
    opt_system_msg = st.text_area(label="System Message", value=opt_system_msg_int, key="system_msg")

with st.sidebar:
    st.divider()
    st.markdown(":blue[**Conversation**]")
    #st.markdown(f"{len(st.session_state.menu_speak_messages)}/{MAX_MESSAGES}")
    uploaded_conversation = st.file_uploader(
        ":green[**Upload Conversation**]",
        type=["json"],
        help="Upload a previously downloaded conversation file"
    )

    if uploaded_conversation is not None:
        if st.button("Load Conversation", type="secondary", icon=":material/upload:", use_container_width=False):
            if upload_conversation(uploaded_conversation):
                st.rerun()

st.markdown("💬 Speech-to-Speech Conversation")

# Initialize session state variables for speech-to-speech functionality
initialize_session_state()


# Session control buttons
col1, col2 = st.columns(2)
with col1:
    if not st.session_state.nova_sonic_session_active:
        if st.button("Start Speech Session", type="primary"):
            # Use asyncio to start the Nova Sonic session
            asyncio.run(start_nova_sonic_session())
            st.rerun()
    else:
        st.success("Session Active")

with col2:
    if st.session_state.nova_sonic_session_active:
        if st.button("End Speech Session", type="secondary"):
            # Use asyncio to end the Nova Sonic session
            asyncio.run(end_nova_sonic_session())
            st.rerun()

# Recording controls
if st.session_state.nova_sonic_session_active:
    st.markdown("### Recording Controls")
    col1, col2 = st.columns(2)

    with col1:
        if not st.session_state.recording:
            if st.button("Start Recording 🎤", type="primary"):
                start_recording()
                st.rerun()
        else:
            st.warning("Recording in progress...")

    with col2:
        if st.session_state.recording:
            if st.button("Stop Recording ⏹️", type="secondary"):
                # Use a thread to stop recording to avoid blocking
                threading.Thread(target=stop_recording).start()
                st.rerun()

# Display conversation history
st.markdown("### Conversation History")
conversation_container = st.container()

with conversation_container:
    for message in st.session_state.conversation_text:
        with st.chat_message(message["role"]):
            st.write(message["content"])

# Text input as an alternative to speech
if st.session_state.nova_sonic_session_active:
    text_input = st.chat_input("Type a message (alternative to speech)")
    if text_input:
        # Add user text to conversation history
        st.session_state.conversation_text.append({"role": "user", "content": text_input})

        # Send text to Nova Sonic
        if st.session_state.nova_client:
            asyncio.run(st.session_state.nova_client.send_text(text_input))

        st.rerun()

# Conversation management buttons
st.markdown(f":violet[**Conversation History: {len(st.session_state.conversation_text)} messages**]")
col1, col2 = st.columns(2)

with col1:
    if st.button("Clear Conversation", type="secondary", icon=":material_clear_all:", use_container_width=False):
        if "conversation_text" in st.session_state:
            st.session_state.conversation_text = []
        st.rerun()


#:markdown/forum:
st.markdown(f"{len(st.session_state.menu_speak_messages)}/{MAX_MESSAGES}")

idx = 1
for msg in st.session_state.menu_speak_messages:
    idx = idx + 1
    contents = msg["content"]
    with st.chat_message(msg["role"]):
        content = contents[0]
        content_text = content["text"]
        document_name = None
        if "user" == msg["role"]:
            if len(contents) > 1:
                content_1 = contents[1]
                if "document" in content_1:
                    content_1_document = content_1["document"]
                    document_name = content_1_document["name"]
            if document_name:
                st.write(f"{content_text} \n\n:green[Document: {document_name}]")
            else:
                st.write(f"{content_text}")
            
            del_idx = idx - 2
            if st.button(f"Delete ({del_idx})", key=f"delete_button_{del_idx}"):
                delete_message_pair(del_idx)
                st.rerun()

        if "assistant" == msg["role"]:
            #assistant_cmd_panel_col1, assistant_cmd_panel_col2, assistant_cmd_panel_col3 = st.columns([0.07,0.23,0.7], gap="small")
            #with assistant_cmd_panel_col2:
            #st.button(key=f"copy_button_{idx}", label='📄', type='primary', on_click=copy_button_clicked, args=[content])
            st.markdown(f"{content_text}")

            # Add feedback mechanism using st.feedback
            feedback_key = f"feedback_{idx}"
            feedback = st.feedback(
                options="stars",
                key=feedback_key,
                on_change=lambda: st.session_state.menu_speak_messages_feedback.update({idx: st.session_state[feedback_key]})
            )

            # Display existing feedback if available
            if idx in st.session_state["menu_speak_messages_feedback"]:
                previous_feedback = st.session_state["menu_speak_messages_feedback"][idx]
                st.info(f"Previous feedback: {previous_feedback + 1} star(s)")

    
if "menu_speak_uploader_key" not in st.session_state:
    st.session_state.menu_speak_uploader_key = 0

#st.write(f"""{opt_fm.is_feature_supported("document_chat")}   {opt_fm.is_feature_supported("vision")}""")

uploaded_file_type_list = []
if opt_fm.is_feature_supported("document_chat"):
    uploaded_file_type_list.extend(["txt", "csv", "pdf", "md"])
if opt_fm.is_feature_supported("vision"):
    uploaded_file_type_list.extend(["png", "jpg", "jpeg"])

if uploaded_file_type_list:
    # #'pdf'|'csv'|'doc'|'docx'|'xls'|'xlsx'|'html'|'txt'|'md',
    uploaded_file = st.file_uploader(
            "Attach Image",
            type=uploaded_file_type_list,
            accept_multiple_files=False,
            label_visibility="collapsed",
            key=f"menu_speak_uploader_key_{st.session_state.menu_speak_uploader_key}"
        )
else:
    uploaded_file = None


prompt = st.chat_input()

uploaded_file_key = None
uploaded_file_name = None
uploaded_file_bytes = None
uploaded_file_type = None
uploaded_file_base64 = None
if uploaded_file:
    if uploaded_file.type in mime_mapping_image: #This field is only supported by Anthropic Claude 3 models.
        uploaded_file_bytes = uploaded_file.read()

        image:Image = Image.open(uploaded_file)
        uploaded_file_name = uploaded_file.name
        uploaded_file_type = uploaded_file.type
        uploaded_file_base64 = image_to_base64(image, mime_mapping[uploaded_file_type])
        st.image(image, caption='upload images', use_container_width=True)
    elif uploaded_file.type in mime_mapping_document:
        uploaded_file_key = uploaded_file.name.replace(".", "_").replace(" ", "_")
        uploaded_file_name = uploaded_file.name
        uploaded_file_type = uploaded_file.type
        bedrock_file_type = mime_mapping_document[uploaded_file_type]
        print(f"-------{bedrock_file_type}")
        if "csv" == bedrock_file_type:
            uploaded_file_bytes = base64.b64encode(uploaded_file.read())
            uploaded_file.seek(0)
            try:
                uploaded_file_df = pd.read_csv(uploaded_file, encoding = "utf-8")
                st.write(uploaded_file_df)
            except Exception as err:
                st.chat_message("system").write(type(err).__name__)
        elif "pdf" == bedrock_file_type:
            uploaded_file_bytes = uploaded_file.read()
            uploaded_file.seek(0)
            st.markdown(uploaded_file_name.replace(".", "_"))
        elif "txt" == bedrock_file_type:
            uploaded_file_bytes = base64.b64encode(uploaded_file.read())
        else:
            st.markdown(uploaded_file_key)
    else:
        print(f"******{uploaded_file.type}") #text/plain

if prompt:
    
    # menu_speak_messages = st.session_state.menu_speak_messages
    # menu_speak_messages_len = len(menu_speak_messages)
    # if menu_speak_messages_len > MAX_MESSAGES:
    #     del menu_speak_messages[0 : (menu_speak_messages_len - MAX_MESSAGES) * 2]
    #st.write(f"""{mime_mapping_image[uploaded_file_type]}""")
    #st.session_state["audio_stream"] = ""

    message_history = st.session_state.menu_speak_messages.copy()
    message_user_latest = {"role": "user", "content": [{ "text": prompt }]}
    if uploaded_file_name:
        content = message_user_latest['content']
        if uploaded_file_type in mime_mapping_image:
            content.append(
                {
                    "image": {
                        "format": mime_mapping_image[uploaded_file_type],
                        "source": {
                            "bytes": uploaded_file_bytes, # If the image dimension is not supported we will get validation error
                        }
                    },
                }
            )
        elif uploaded_file.type in mime_mapping_document:
            #uploaded_file_name_clean = str(uuid.uuid4()) #uploaded_file_name.replace(".", "_").replace(" ", "_")
            uploaded_file_name_clean = uploaded_file_key
            content.append(
                {
                    "document": {
                        "format": mime_mapping_document[uploaded_file_type],
                        "name": uploaded_file_name_clean, #uploaded_file_key
                        "source": {
                            "bytes": uploaded_file_bytes,
                        }
                    },
                }
            )
        else:
            st.write(f"Not supported file type: {uploaded_file.type}")
    message_history.append(message_user_latest)
    #print(f"******{message_user_latest}")
    st.chat_message("user").write(prompt)

    system_prompts = [{"text" : opt_system_msg}]
    
    inference_config = {
        "temperature": opt_temperature,
        "maxTokens": opt_max_tokens,
        "topP": opt_top_p,
        #stopSequences 
    }

    additional_model_fields = {}

    if opt_fm_top_k.is_supported:
        additional_model_fields[opt_fm_top_k.name] = opt_top_k
    
    # If additional_model_fields is an empty dictionary, set it to None
    if additional_model_fields == {}:
        additional_model_fields = None

    # additional_model_fields = {"top_k": opt_top_k}
    # if opt_model_id.startswith("cohere"):
    #     additional_model_fields = None
    # if opt_model_id.startswith("meta") or opt_model_id.startswith("us.meta"):
    #     additional_model_fields = None
    # if opt_model_id.startswith("mistral"):
    #     additional_model_fields = None



    #print(json.dumps(inference_config, indent=3))
    #print(json.dumps(system_prompts, indent=3))

    with st.spinner('Processing...'):

        try:
            
            if "anthropic.claude-3-5-sonnet-20241022-v2:0" == opt_model_id or "us.anthropic.claude-3-5-sonnet-20241022-v2:0" == opt_model_id:
                    response = bedrock_runtime_us_west_2.converse_stream(
                    modelId=opt_model_id,
                    messages=message_history,
                    system=system_prompts,
                    inferenceConfig=inference_config,
                    additionalModelRequestFields=additional_model_fields
                )
            else:
                response = bedrock_runtime.converse_stream(
                    modelId=opt_model_id,
                    messages=message_history,
                    system=system_prompts,
                    inferenceConfig=inference_config,
                    additionalModelRequestFields=additional_model_fields
                )
            

            #with st.chat_message("assistant", avatar=setAvatar("assistant")):
            result_text = ""
            with st.chat_message("assistant"):
                result_container = st.container(border=True)
                result_area = st.empty()
                stream = response.get('stream')
                for event in stream:
                    
                    if 'messageStart' in event:
                        #opts = f"| temperature={opt_temperature} top_p={opt_top_p} top_k={opt_top_k} max_tokens={opt_max_tokens} role= {event['messageStart']['role']}"
                        #result_container.write(opts)                    
                        pass

                    if 'contentBlockDelta' in event:
                        text = event['contentBlockDelta']['delta']['text']
                        result_text += f"{text}"
                        result_area.write(result_text)

                    if 'messageStop' in event:
                        #'stopReason': 'end_turn'|'tool_use'|'max_tokens'|'stop_sequence'|'content_filtered'
                        stop_reason = event['messageStop']['stopReason']
                        if stop_reason == 'end_turn':
                            pass
                        else:
                            stop_reason_display = stop_reason
                            if stop_reason == 'max_tokens':
                                stop_reason_display = "Insufficient Tokens. Increaes MaxToken Settings."
                            result_text_error = f"{result_text}\n\n:red[Generation Stopped: {stop_reason_display}]"
                            result_area.write(result_text_error)

                    if 'metadata' in event:
                        metadata = event['metadata']
                        if 'usage' in metadata:
                            input_token_count = metadata['usage']['inputTokens']
                            output_token_count = metadata['usage']['outputTokens']
                            total_token_count = metadata['usage']['totalTokens']
                        if 'metrics' in event['metadata']:
                            latency = metadata['metrics']['latencyMs']
                        stats = f"| token.in={input_token_count} token.out={output_token_count} token={total_token_count} latency={latency} provider={opt_fm.provider}"
                        result_container.write(stats)
                        user_name = "default_user"  # Replace with actual user name if available
                        try:
                            push_to_cloudwatch(user_name, input_token_count, output_token_count, total_token_count)
                        except Exception as e:
                            logger.error(f"Failed to push logs to CloudWatch: {str(e)}")

                    if "internalServerException" in event:
                        exception = event["internalServerException"]
                        result_text += f"\n\{exception}"
                        result_area.write(result_text)
                    if "modelStreamErrorException" in event:
                        exception = event["modelStreamErrorException"]
                        result_text += f"\n\{exception}"
                        result_area.write(result_text)
                    if "throttlingException" in event:
                        exception = event["throttlingException"]
                        result_text += f"\n\{exception}"
                        result_area.write(result_text)
                    if "validationException" in event:
                        exception = event["validationException"]
                        result_text += f"\n\{exception}"
                        result_area.write(result_text)

                #col1, col2, col3 = st.columns([1,1,5])

                #with col1:
                    #st.button(key='copy_button', label='📄', type='primary', on_click=copy_button_clicked, args=[result_text])
                #    pass
                #with col2:
                #    if "audio_stream" not in st.session_state or st.session_state["audio_stream"] == "":
                #        st.button(key='recite_button', label='▶️', type='primary', on_click=recite_button_clicked, args=[result_text])
                #with col3:
                #    #st.markdown('3')
                #    pass
            
            message_assistant_latest = {"role": "assistant", "content": [{ "text": result_text }]}

            st.session_state.menu_speak_messages.append(message_user_latest)
            st.session_state.menu_speak_messages.append(message_assistant_latest)

            
            # Trim message History
            menu_speak_messages = st.session_state.menu_speak_messages
            menu_speak_messages_len = len(menu_speak_messages)
            if menu_speak_messages_len > MAX_MESSAGES:
                del menu_speak_messages[0 : (menu_speak_messages_len - MAX_MESSAGES) * 2] #make sure we remove both the user and assistant responses
            #print(f"menu_speak_messages_len={menu_speak_messages_len}")

            if uploaded_file_name:
                st.session_state.menu_speak_uploader_key += 1

            #print(json.dumps(message_user_latest, indent=2))
            #print(message_user_latest)
            #st.rerun()

        except ClientError as err:
            message = err.response["Error"]["Message"]
            logger.error("A client error occurred: %s", message)
            print("A client error occured: " + format(message))
            st.chat_message("system").write(message)
        except ReadTimeoutError as err:
            logger.error("A client error occurred: %s", err)  # Log the error directly
            print("A client error occurred: " + str(err))     # Print the error as a string
            st.chat_message("system").write(str(err))         # Use str() to display in chat message


#if "audio_stream" in st.session_state and st.session_state["audio_stream"] != "":
#    audio_bytes = BytesIO(st.session_state['audio_stream'])
#    st.audio(audio_bytes, format='audio/mp3', autoplay=False)

#st.markdown(f"Entry: 💬 {len(st.session_state.menu_speak_messages)}/{MAX_MESSAGES}")
st.markdown(f":violet[**Entry: {len(st.session_state.menu_speak_messages)}/{MAX_MESSAGES}**]")
col1, col2, col3 = st.columns(3)

with col1:
    #st.markdown("Conversation: ")
    if st.button("Download Conversation", type="secondary", icon=":material/download:", use_container_width=False):
        if not st.session_state.menu_speak_messages:
            st.error("No data to download!")
        else:
            try:
                conversation_file = get_conversation_download()
                st.download_button(
                    label="Download JSON",
                    data=conversation_file,
                    file_name=f"conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    key="download_button"
                )
            except Exception as e:
                st.error(f"Error downloading conversation: {str(e)}")

with col2:
    # if st.button("Upload JSON File"):
    #     uploaded_file = st.file_uploader("Choose a JSON file", type=["json"], key="json_uploader")
    #     if uploaded_file is not None:
    #         upload_conversation(uploaded_file)
    #         st.rerun()
    
    #st.markdown(f"💬 {len(st.session_state.menu_speak_messages)}/{MAX_MESSAGES}")
    #st.info(f"{len(st.session_state.menu_speak_messages)}/{MAX_MESSAGES}", icon=":material/forum:")
    pass
with col3:
    if st.button("Clear Conversation", type="secondary", icon=":material/clear_all:", use_container_width=False):
        clear_conversation()
        #st.success("Conversation cleared successfully!")
        st.rerun()