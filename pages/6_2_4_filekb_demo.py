import boto3
import streamlit as st
import cmn_auth
import cmn_settings
import logging
from botocore.exceptions import ClientError

# https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/bedrock-agent-runtime/client/retrieve_and_generate.html

# ValidationException: An error occurred (ValidationException) when calling the RetrieveAndGenerate operation: Token limit exceeded: 
# input contains 26105 tokens and exceeds limit of 

# Error 2 - Haiku is not supported
#The provided model is not supported for EXTERNAL_SOURCES RetrieveAndGenerateType. Update the model arn then retry your request.

###### AUTH START #####

#if not cmn_auth.check_password():
#   st.stop()

######  AUTH END #####

AWS_REGION = cmn_settings.AWS_REGION
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
client = boto3.client("bedrock-agent-runtime", region_name=AWS_REGION)


st.set_page_config(
    page_title="File Query",
    page_icon="📓",
    layout="centered", # "centered" or "wide"
    initial_sidebar_state="expanded", #"auto", "expanded", or "collapsed"
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': None
    }
)

st.logo(icon_image="images/logo.png", image="images/logo_text.png")

opt_model_id_list = [
    "anthropic.claude-3-sonnet-20240229-v1:0"
]

with st.sidebar:
    st.markdown(":blue[Settings]")
    opt_model_id = st.selectbox(label="Model ID", options=opt_model_id_list, index = 0, key="model_id")

st.title("File Query v2.4")
st.write("Knowledge Bases on File Upload for Amazon Bedrock")

if "menu_filequery_messages" not in st.session_state:
    st.session_state.menu_filequery_messages = []

for message in st.session_state.menu_filequery_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "citations" in message:
            with st.expander("citations"):
                st.json(message["citations"])


if "menu_filequery_session_id" in st.session_state:
    menu_filequery_session_id = st.session_state["menu_filequery_session_id"]
    st.markdown(f":blue[Session: {menu_filequery_session_id}]")

#### 
uploaded_file = st.file_uploader(
    "Maximum of 20000 Tokens",
    type=["PDF", "MD", "TXT", "DOCX", "HTML", "CSV", "XLS", "XLSX"],
    accept_multiple_files=False,
)

if uploaded_file:
    uploaded_file_bytes = uploaded_file.getvalue()
    uploaded_file_name = uploaded_file.name
    uploaded_file_type = uploaded_file.type

#### 

if prompt := st.chat_input(
    "Ask question about the file." if uploaded_file else "Choose and upload a file first",
    disabled=(not uploaded_file),
):
    st.session_state.menu_filequery_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    
    with st.chat_message("assistant"):
        
        try:

            params = {
                "input": { "text": prompt },
                "retrieveAndGenerateConfiguration": {
                    "type": "EXTERNAL_SOURCES",
                    "externalSourcesConfiguration": {
                        "modelArn": opt_model_id,
                        "sources": [
                            {
                                "sourceType": "BYTE_CONTENT",
                                "byteContent": {
                                    "contentType": uploaded_file_type,
                                    "data": uploaded_file_bytes,
                                    "identifier": uploaded_file_name,
                                },
                            }
                        ],
                    },
                },
            }

            if "menu_filequery_session_id" in st.session_state:
                params["sessionId"] = st.session_state["menu_filequery_session_id"]

            response = client.retrieve_and_generate(**params)

            st.markdown(response["output"]["text"])
            with st.expander("citations"):
                st.json(response["citations"])

            st.session_state["menu_filequery_session_id"] = response["sessionId"]
            st.session_state.menu_filequery_messages.append(
                {
                    "role": "assistant",
                    "content": response["output"]["text"],
                    "citations": response["citations"],
                }
            )

        except ClientError as err:
            message = err.response["Error"]["Message"]
            logger.error("A client error occurred: %s", message)
            st.markdown(f":red[{message}]")
            #st.chat_message("system").write(message)