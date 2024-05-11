import boto3
import streamlit as st
import cmn_auth
import settings

###### AUTH START #####

if not cmn_auth.check_password():
   st.stop()

######  AUTH END #####

AWS_REGION = settings.AWS_REGION
client = boto3.client("bedrock-agent-runtime", region_name=AWS_REGION)
#bedrock_runtime = boto3.client('bedrock-runtime', region_name=AWS_REGION)

st.title("KB - Inquire Uploaded Document")
st.write("Knowledge Bases for Amazon Bedrock")

uploaded_file = st.file_uploader(
    "The supported file types are PDF, MD, TXT, DOCX, HTML, CSV, XLS, and XLSX",
    type=["PDF", "MD", "TXT", "DOCX", "HTML", "CSV", "XLS", "XLSX"],
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
    "何でも聞いて下さい" if uploaded_file else "ファイルを選択してください",
    disabled=(not uploaded_file),
):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):

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
