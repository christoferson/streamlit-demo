import streamlit as st
import boto3
import cmn_settings
import json
import logging
import cmn_auth
import pyperclip
import os
from io import BytesIO
import sys
import subprocess
from contextlib import closing
from tempfile import gettempdir

from pydub import AudioSegment
from pydub.playback import play

from botocore.exceptions import BotoCoreError, ClientError

AWS_REGION = cmn_settings.AWS_REGION

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/bedrock-runtime/client/converse_stream.html

###### AUTH START #####

#if not cmn_auth.check_password():
#   st.stop()

######  AUTH END #####

####################################################################################

bedrock_runtime = boto3.client('bedrock-runtime', region_name=AWS_REGION)
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
    }
)

st.logo(icon_image="images/logo.png", image="images/logo_text.png")

st.markdown(
    """
    <style>
    button[kind="primary"] {
        background: none!important;
        border: none;
        padding: 0!important;
        margin: 0;
        color: black !important;
        text-decoration: none;
        cursor: pointer;
        border: none !important;
    }
    button[kind="primary"]:hover {
        text-decoration: none;
        color: black !important;
    }
    button[kind="primary"]:focus {
        outline: none !important;
        box-shadow: none !important;
        color: black !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Play Audio Button stAudio
st.markdown(
    """
    <style>
    #.stAudio {
    #    max-width: 70px;
    #    max-height: 50px;
    #}
    #audio::-webkit-media-controls-time-remaining-display,
    #audio::-webkit-media-controls-current-time-display {
    #    max-width: 50%;
    #    max-height: 20px;
    #}
    </style>
    """,
    unsafe_allow_html=True,
)


def copy_button_clicked(text):
    pyperclip.copy(text)
    #st.session_state.button = not st.session_state.button

def recite_button_clicked(text):
    try:
        # Request speech synthesis
        response = polly.synthesize_speech(Text=text, OutputFormat="mp3", VoiceId="Joanna")
    except (BotoCoreError, ClientError) as error:
        print(error)
        return

    if "AudioStream" in response:
        # Note: Closing the stream is important because the service throttles on the
        # number of parallel connections. Here we are using contextlib.closing to
        # ensure the close method of the stream object will be called automatically
        # at the end of the with statement's scope.
            with closing(response["AudioStream"]) as stream:
                output = os.path.join(gettempdir(), "speech.mp3")
                try:
                    # Open a file for writing the output as a binary stream
                    sound = stream.read()
                    with open(output, "wb") as file:
                        file.write(sound)
                    
                    st.session_state['audio_stream'] = sound

                    #data = open(output, 'rb').read()
                    #song = AudioSegment.from_file(BytesIO(data), format="mp3")
                    #play(song)
                except IOError as error:
                    # Could not write to file, exit gracefully
                    print(error)
                    #sys.exit(-1)
                    st.session_state['audio_stream'] = ""
                    return
                
                print("**********************************************************************")
                try:                 
                    print(f"/n/n---------------------------------------------------------/n{output}")   
                    #data = open(output, 'rb').read()
                    #print("------")
                    #song = AudioSegment.from_file(BytesIO(data), format="mp3")
                    #print("------")
                    #play(song)
                except IOError as error:
                    print(error)
                    return                

    else:
        # The response didn't contain audio data, exit gracefully
        print("Could not stream audio")
        #sys.exit(-1)
        return
       

opt_model_id_list = [
    "anthropic.claude-3-sonnet-20240229-v1:0",
    "anthropic.claude-3-haiku-20240307-v1:0"
]

with st.sidebar:
    opt_model_id = st.selectbox(label="Model ID", options=opt_model_id_list, index = 0, key="model_id")
    opt_temperature = st.slider(label="Temperature", min_value=0.0, max_value=1.0, value=0.1, step=0.1, key="temperature")
    opt_top_p = st.slider(label="Top P", min_value=0.0, max_value=1.0, value=1.0, step=0.1, key="top_p")
    opt_top_k = st.slider(label="Top K", min_value=0, max_value=500, value=250, step=1, key="top_k")
    opt_max_tokens = st.slider(label="Max Tokens", min_value=0, max_value=4096, value=2048, step=1, key="max_tokens")
    opt_system_msg = st.text_area(label="System Message", value="You are a question and answering chatbot", key="system_msg")



st.title("💬 Converse 3-5-1")
st.write("Ask LLM Questions")

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        #{"role": "user", "content": "Hello there."},
        #{"role": "assistant", "content": "How can I help you?"}
    ]

#if "audio_stream" not in st.session_state:
#    st.session_state["audio_stream"] = ""

idx = 1
for msg in st.session_state.messages:
    idx = idx + 1
    contents = msg["content"]
    with st.chat_message(msg["role"]):
        content = contents[0]["text"]
        st.write(content)
        if "assistant" == msg["role"]:
            #assistant_cmd_panel_col1, assistant_cmd_panel_col2, assistant_cmd_panel_col3 = st.columns([0.07,0.23,0.7], gap="small")
            #with assistant_cmd_panel_col2:
            st.button(key=f"copy_button_{idx}", label='📄', type='primary', on_click=copy_button_clicked, args=[content])


if prompt := st.chat_input():
    
    st.session_state["audio_stream"] = ""

    message_history = st.session_state.messages.copy()
    message_user_latest = {"role": "user", "content": [{ "text": prompt }]}
    message_history.append(message_user_latest)
    st.chat_message("user").write(prompt)

    system_prompts = [{"text" : opt_system_msg}]
    
    inference_config = {
        "temperature": opt_temperature,
        "maxTokens": opt_max_tokens,
        "topP": opt_top_p,
    }
    additional_model_fields = {"top_k": opt_top_k}
    #print(json.dumps(inference_config, indent=3))
    #print(json.dumps(system_prompts, indent=3))

    try:
        
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
                    opts = f"| temperature={opt_temperature} top_p={opt_top_p} top_k={opt_top_k} max_tokens={opt_max_tokens} role= {event['messageStart']['role']}"
                    result_container.write(opts)                    

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
                    stats = f"| token.in={input_token_count} token.out={output_token_count} token={total_token_count} latency={latency}"
                    result_container.write(stats)

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

            col1, col2, col3 = st.columns([1,1,5])

            with col1:
                st.button(key='copy_button', label='📄', type='primary', on_click=copy_button_clicked, args=[result_text])
            with col2:
                if "audio_stream" not in st.session_state or st.session_state["audio_stream"] == "":
                    st.button(key='recite_button', label='▶️', type='primary', on_click=recite_button_clicked, args=[result_text])
            with col3:
                st.markdown('3')
        
        message_assistant_latest = {"role": "assistant", "content": [{ "text": result_text }]}
        st.session_state.messages.append(message_user_latest)
        st.session_state.messages.append(message_assistant_latest)

    except ClientError as err:
        message = err.response["Error"]["Message"]
        logger.error("A client error occurred: %s", message)
        print("A client error occured: " + format(message))
        st.chat_message("system").write(message)

if "audio_stream" in st.session_state and st.session_state["audio_stream"] != "":
    audio_bytes = BytesIO(st.session_state['audio_stream'])
    st.audio(audio_bytes, format='audio/mp3', autoplay=False)