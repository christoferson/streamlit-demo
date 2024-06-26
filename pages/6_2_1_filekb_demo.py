import boto3
import streamlit as st
import cmn_auth
import cmn_settings
import logging
from botocore.exceptions import ClientError

#ValidationException: An error occurred (ValidationException) when calling the RetrieveAndGenerate operation: Token limit exceeded: 
# input contains 26105 tokens and exceeds limit of 20000

###### AUTH START #####

#if not cmn_auth.check_password():
#   st.stop()

######  AUTH END #####

AWS_REGION = cmn_settings.AWS_REGION
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
client = boto3.client("bedrock-agent-runtime", region_name=AWS_REGION)

st.title("File Query v2 1")
st.write("Knowledge Bases on File Upload for Amazon Bedrock")

uploaded_file = st.file_uploader(
    "The supported file types are PDF, MD, TXT, DOCX, HTML, CSV, XLS, and XLSX",
    type=["PDF", "MD", "TXT", "DOCX", "HTML", "CSV", "XLS", "XLSX"],
    accept_multiple_files=False,
)

if uploaded_file:
    uploaded_file_bytes = uploaded_file.getvalue()
    uploaded_file_name = uploaded_file.name
    uploaded_file_type = uploaded_file.type

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "citations" in message:
            with st.expander("citations"):
                st.json(message["citations"])

if prompt := st.chat_input(
    "Ask question about the file." if uploaded_file else "Choose and upload a file first",
    disabled=(not uploaded_file),
):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    
    with st.chat_message("assistant"):
        
        try:

            params = {
                "input": {"text": prompt},
                "retrieveAndGenerateConfiguration": {
                    "type": "EXTERNAL_SOURCES",
                    "externalSourcesConfiguration": {
                        "modelArn": "anthropic.claude-3-sonnet-20240229-v1:0",
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

            if "sessionId" in st.session_state:
                params["sessionId"] = st.session_state["sessionId"]

            response = client.retrieve_and_generate(**params)

            st.markdown(response["output"]["text"])
            with st.expander("citations"):
                st.json(response["citations"])

            st.session_state["sessionId"] = response["sessionId"]
            st.session_state.messages.append(
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