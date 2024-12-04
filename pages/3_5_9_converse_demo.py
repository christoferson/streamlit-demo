import streamlit as st
import boto3
import cmn_settings
import cmn_constants
import json
import logging
import cmn_auth
import os
from io import BytesIO

from PIL import Image
import io
import base64
import pandas as pd
from cmn.bedrock_models import FoundationModel
from datetime import datetime

from botocore.exceptions import BotoCoreError, ClientError, ReadTimeoutError

AWS_REGION = cmn_settings.AWS_REGION
MAX_MESSAGES = 80 * 2

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

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


# st.markdown(
#         """
#         <style>
#         .stAppViewMain {
#             color: white;
#             background-color: #1E1E1E;
#         }
#         .stChatMessage {
#             color: white;
#             background-color: #1E1E1E;
#         }
#         .stVerticalBlockBorderWrapper {
#             color: white;
#             background-color: #1E1E1E;
#         }
#         .stCode {
#             color: white;
#             background-color: #1E1E1E;
#         }
#         .stMarkdown {
#             color: white;
#         }
#         .stButton>button {
#             color: #4CAF50;
#             background-color: #2E2E2E;
#             border: 2px solid #4CAF50;
#         }
#         </style>
#         """,
#         unsafe_allow_html=True
#     )

def image_to_base64(image,mime_type:str):
    buffer = io.BytesIO()
    image.save(buffer, format=mime_type)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def get_conversation_download():
    conversation = st.session_state.menu_converse_messages
    conversation_json = json.dumps(conversation, indent=2)
    return BytesIO(conversation_json.encode())

def clear_conversation():
    st.session_state.menu_converse_messages = []
    st.session_state.menu_converse_uploader_key = 0


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

def upload_conversation(uploaded_file):
    try:
        # Read the uploaded JSON file
        content = uploaded_file.getvalue().decode("utf-8")
        conversation = json.loads(content)

        # Validate the structure of the uploaded conversation
        if not isinstance(conversation, list):
            raise ValueError("Invalid conversation format: expected a list")

        for message in conversation:
            if not isinstance(message, dict) or "role" not in message or "content" not in message:
                raise ValueError("Invalid message format: expected 'role' and 'content' keys")

            if message["role"] not in ["user", "assistant", "system"]:
                raise ValueError(f"Invalid role: {message['role']}")

            if not isinstance(message["content"], list):
                raise ValueError("Invalid content format: expected a list")

            for content_item in message["content"]:
                if not isinstance(content_item, dict) or "text" not in content_item:
                    raise ValueError("Invalid content item: expected 'text' key")

        # If validation passes, update the session state
        st.session_state.menu_converse_messages = conversation
        #st.success("Conversation uploaded successfully!")
        #st.rerun()

    except json.JSONDecodeError:
        st.error("Invalid JSON file. Please upload a valid JSON conversation file.")
    except ValueError as e:
        st.error(f"Error in conversation structure: {str(e)}")
    except Exception as e:
        st.error(f"An error occurred while processing the file: {str(e)}")

def delete_message_pair(index):
    if index < len(st.session_state.menu_converse_messages):
        del st.session_state.menu_converse_messages[index:index+2]
    if f"feedback_{index+1}" in st.session_state.menu_converse_messages_feedback:
        del st.session_state.menu_converse_messages_feedback[f"feedback_{index+1}"]

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

    "us.anthropic.claude-3-5-sonnet-20241022-v2:0",

    "anthropic.claude-3-5-sonnet-20240620-v1:0",
    "anthropic.claude-3-sonnet-20240229-v1:0",
    "anthropic.claude-3-haiku-20240307-v1:0",
    "anthropic.claude-3-5-sonnet-20241022-v2:0",
    #"anthropic.claude-3-opus-20240229-v1:0",
    "us.anthropic.claude-3-haiku-20240307-v1:0",
    "us.anthropic.claude-3-sonnet-20240229-v1:0",
    "us.anthropic.claude-3-opus-20240229-v1:0",
    "us.anthropic.claude-3-5-sonnet-20240620-v1:0",
    "us.anthropic.claude-3-5-sonnet-20241022-v2:0",

    "amazon.nova-pro-v1:0",
    
    "cohere.command-r-v1:0", # The model returned the following errors: Malformed input request: #: extraneous key [top_k] is not permitted, please reformat your input and try again.
    "cohere.command-r-plus-v1:0",
    "meta.llama2-13b-chat-v1", # Llama 2 Chat 13B
    "meta.llama2-70b-chat-v1", # Llama 2 Chat 70B
    "meta.llama3-8b-instruct-v1:0", # Llama 3 8b Instruct
    "meta.llama3-70b-instruct-v1:0",  # Llama 3 70b Instruct
    "us.meta.llama3-2-11b-instruct-v1:0", # Vision
    "us.meta.llama3-2-90b-instruct-v1:0", # Vision
    #"mistral.mistral-7b-instruct-v0:2", # Mistral 7B Instruct Does not support system message
    #"mistral.mixtral-8x7b-instruct-v0:1", # Mixtral 8X7B Instruct Does not support system message
    "mistral.mistral-small-2402-v1:0", # Mistral Small
    "mistral.mistral-large-2402-v1:0", # Mistral Large
]

# "You are a question and answering chatbot"
# - Maintaining a helpful and courteous tone throughout the interaction.
opt_system_msg_int = """You are a smart and helpful Assistant. Your tasks include:
- Providing detailed, step-by-step, and professional responses to user queries.
- Ensuring accuracy and depth in your explanations.
- Adapting your language and complexity to suit the user's level of understanding.
- Offering relevant examples or analogies to clarify complex concepts.
- Citing reliable sources when appropriate to support your information.
Please think through each step carefully before responding to ensure clarity, completeness, and coherence in your answer. 
If any part of the query is unclear, don't hesitate to ask for clarification to provide the most accurate and helpful response possible.
"""

with st.sidebar:
    opt_model_id = st.selectbox(label="Model ID", options=opt_model_id_list, index = 0, key="model_id")

opt_fm:FoundationModel = FoundationModel.find(opt_model_id)

opt_fm_max_tokens = opt_fm.InferenceParameter.get("MaxTokensToSample")
opt_fm_top_k = opt_fm.InferenceParameter.get("TopK")

with st.sidebar:
    opt_temperature = st.slider(label="Temperature", min_value=0.0, max_value=1.0, value=0.1, step=0.1, key="temperature")
    opt_top_p = st.slider(label="Top P", min_value=0.0, max_value=1.0, value=1.0, step=0.1, key="top_p")
    if opt_fm_top_k.isSupported():
        opt_top_k = st.slider(label="Top K", min_value=0, max_value=500, value=250, step=1, key="top_k")
    else:
        opt_top_k = 0
    opt_max_tokens = st.slider(label="Max Tokens", min_value=opt_fm_max_tokens.MinValue, max_value=opt_fm_max_tokens.MaxValue, value=opt_fm_max_tokens.DefaultValue, step=1, key="max_tokens")
    opt_system_msg = st.text_area(label="System Message", value=opt_system_msg_int, key="system_msg")


st.markdown("💬 Converse 3-5-3")

if "menu_converse_messages" not in st.session_state:
    st.session_state["menu_converse_messages"] = []

# Add this to store feedback
if "menu_converse_messages_feedback" not in st.session_state:
    st.session_state["menu_converse_messages_feedback"] = {}

st.markdown(f"{len(st.session_state.menu_converse_messages)}/{MAX_MESSAGES}")

idx = 1
for msg in st.session_state.menu_converse_messages:
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
            st.write(f"{content_text} \n\n:green[Document: {document_name}]")

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
                on_change=lambda: st.session_state.menu_converse_messages_feedback.update({idx: st.session_state[feedback_key]})
            )

            # Display existing feedback if available
            if idx in st.session_state["menu_converse_messages_feedback"]:
                previous_feedback = st.session_state["menu_converse_messages_feedback"][idx]
                st.info(f"Previous feedback: {previous_feedback + 1} star(s)")

    
if "menu_converse_uploader_key" not in st.session_state:
    st.session_state.menu_converse_uploader_key = 0

#st.write(f"""{opt_fm.isFeatureSupported("document_chat")}   {opt_fm.isFeatureSupported("vision")}""")

uploaded_file_type_list = []
if opt_fm.isFeatureSupported("document_chat"):
    uploaded_file_type_list.extend(["txt", "csv", "pdf", "md"])
if opt_fm.isFeatureSupported("vision"):
    uploaded_file_type_list.extend(["png", "jpg", "jpeg"])

if uploaded_file_type_list:
    # #'pdf'|'csv'|'doc'|'docx'|'xls'|'xlsx'|'html'|'txt'|'md',
    uploaded_file = st.file_uploader(
            "Attach Image",
            type=uploaded_file_type_list,
            accept_multiple_files=False,
            label_visibility="collapsed",
            key=f"menu_converse_uploader_key_{st.session_state.menu_converse_uploader_key}"
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
        st.image(image, caption='upload images', use_column_width=True)
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
    
    # menu_converse_messages = st.session_state.menu_converse_messages
    # menu_converse_messages_len = len(menu_converse_messages)
    # if menu_converse_messages_len > MAX_MESSAGES:
    #     del menu_converse_messages[0 : (menu_converse_messages_len - MAX_MESSAGES) * 2]
    #st.write(f"""{mime_mapping_image[uploaded_file_type]}""")
    #st.session_state["audio_stream"] = ""

    message_history = st.session_state.menu_converse_messages.copy()
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

    if opt_fm_top_k.isSupported():
        additional_model_fields[opt_fm_top_k.Name] = opt_top_k
    
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

            st.session_state.menu_converse_messages.append(message_user_latest)
            st.session_state.menu_converse_messages.append(message_assistant_latest)

            
            # Trim message History
            menu_converse_messages = st.session_state.menu_converse_messages
            menu_converse_messages_len = len(menu_converse_messages)
            if menu_converse_messages_len > MAX_MESSAGES:
                del menu_converse_messages[0 : (menu_converse_messages_len - MAX_MESSAGES) * 2] #make sure we remove both the user and assistant responses
            #print(f"menu_converse_messages_len={menu_converse_messages_len}")

            if uploaded_file_name:
                st.session_state.menu_converse_uploader_key += 1

            #print(json.dumps(message_user_latest, indent=2))
            #print(message_user_latest)

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


col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Download Conversation"):
        conversation_file = get_conversation_download()
        st.download_button(
            label="Download JSON",
            data=conversation_file,
            file_name="conversation.json",
            mime="application/json"
        )

with col2:
    # if st.button("Upload JSON File"):
    #     uploaded_file = st.file_uploader("Choose a JSON file", type=["json"], key="json_uploader")
    #     if uploaded_file is not None:
    #         upload_conversation(uploaded_file)
    #         st.rerun()
    pass

with col3:
    if st.button("Clear Conversation"):
        clear_conversation()
        st.rerun()