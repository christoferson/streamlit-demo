import streamlit as st
import boto3
import cmn_settings
import cmn_constants
import json
import logging
import cmn_auth
import os
from io import BytesIO
import sys
import subprocess
from contextlib import closing
from tempfile import gettempdir

from pydub import AudioSegment
from pydub.playback import play
from PIL import Image
import io
import base64
import uuid
import pandas as pd
from cmn.bedrock_models import FoundationModel

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

st.markdown(cmn_constants.css_button_primary, unsafe_allow_html=True)

def image_to_base64(image,mime_type:str):
    buffer = io.BytesIO()
    image.save(buffer, format=mime_type)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

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
    "anthropic.claude-3-5-sonnet-20240620-v1:0",
    "anthropic.claude-3-sonnet-20240229-v1:0",
    "anthropic.claude-3-haiku-20240307-v1:0",
    #"anthropic.claude-3-opus-20240229-v1:0",
    "us.anthropic.claude-3-haiku-20240307-v1:0",
    "us.anthropic.claude-3-sonnet-20240229-v1:0",
    "us.anthropic.claude-3-opus-20240229-v1:0",
    "us.anthropic.claude-3-5-sonnet-20240620-v1:0",
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
    opt_temperature = st.slider(label="Temperature", min_value=0.0, max_value=1.0, value=0.1, step=0.1, key="temperature")
    opt_top_p = st.slider(label="Top P", min_value=0.0, max_value=1.0, value=1.0, step=0.1, key="top_p")
    opt_top_k = st.slider(label="Top K", min_value=0, max_value=500, value=250, step=1, key="top_k")
    opt_max_tokens = st.slider(label="Max Tokens", min_value=0, max_value=4096, value=2048, step=1, key="max_tokens")
    opt_system_msg = st.text_area(label="System Message", value=opt_system_msg_int, key="system_msg")


opt_fm:FoundationModel = FoundationModel.find(opt_model_id)

st.markdown("💬 Converse 3-5-3")

if "menu_converse_messages" not in st.session_state:
    st.session_state["menu_converse_messages"] = []


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
            st.markdown(f"{content_text} \n\n:green[Document: {document_name}]")
        if "assistant" == msg["role"]:
            #assistant_cmd_panel_col1, assistant_cmd_panel_col2, assistant_cmd_panel_col3 = st.columns([0.07,0.23,0.7], gap="small")
            #with assistant_cmd_panel_col2:
            #st.button(key=f"copy_button_{idx}", label='📄', type='primary', on_click=copy_button_clicked, args=[content])
            st.markdown(f"{content_text}")
    
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
    st.session_state["audio_stream"] = ""

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

    additional_model_fields = {"top_k": opt_top_k}
    if opt_model_id.startswith("cohere"):
        additional_model_fields = None
    if opt_model_id.startswith("meta") or opt_model_id.startswith("us.meta"):
        additional_model_fields = None
    if opt_model_id.startswith("mistral"):
        additional_model_fields = None



    #print(json.dumps(inference_config, indent=3))
    #print(json.dumps(system_prompts, indent=3))

    with st.spinner('Processing...'):

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


if "audio_stream" in st.session_state and st.session_state["audio_stream"] != "":
    audio_bytes = BytesIO(st.session_state['audio_stream'])
    st.audio(audio_bytes, format='audio/mp3', autoplay=False)