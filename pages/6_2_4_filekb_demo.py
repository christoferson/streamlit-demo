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
runtime_client = boto3.client("bedrock-runtime", region_name=AWS_REGION)

# RetrieveAndGenerate EXTERNAL_SOURCES rejects images ("Unsupported file
# format"), so image attachments are routed to the Converse API instead.
_IMAGE_FORMATS = {
    "image/png":  "png",
    "image/jpeg": "jpeg",
    "image/gif":  "gif",
    "image/webp": "webp",
}


st.set_page_config(
    page_title="File Query",
    page_icon= ":bulb:", #"📓",
    layout="centered", # "centered" or "wide"
    initial_sidebar_state="expanded", #"auto", "expanded", or "collapsed"
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': None
    }
)

st.logo(icon_image="images/logo.png", image="images/logo_text.png")

#stSidebarHeader stSidebarNav stSidebarNavSeparator
st.html(
    """
<style>
[data-testid="stSidebarContent"] {
    color: white;
    background-color: none;
}
[data-testid="stSidebarNav"] {
    color: white;
    background-color: none;
}
[data-testid="stSidebarNavItems"] {
    color: white;
    background-color: none;
    scrollbar-color: lightgray lightblue;
    overflow-y: scroll;
}

[data-testid="stSidebarNavSeparator"] {
    color: white;
    background-color: none;
    
}

</style>
"""
)

opt_model_id_list = [
    "global.anthropic.claude-sonnet-4-5-20250929-v1:0",
    #"anthropic.claude-3-sonnet-20240229-v1:0"
]

opt_top_k = 250
opt_top_p = 1.0

with st.sidebar:
    #st.markdown(":blue[Settings]")
    opt_model_id = st.selectbox(label="Model ID", options=opt_model_id_list, index = 0, key="model_id")
    opt_temperature = st.slider(label="Temperature", min_value=0.0, max_value=1.0, value=0.0, step=0.1, key="temperature")
    #opt_top_p = st.slider(label="Top P", min_value=0.0, max_value=1.0, value=1.0, step=0.1, key="top_p")
    #opt_top_k = st.slider(label="Top K", min_value=0, max_value=500, value=250, step=1, key="top_k")
    opt_max_tokens = st.slider(label="Max Tokens", min_value=0, max_value=4096, value=2048, step=1, key="max_tokens")

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
    menu_filequery_uploaded_file_name = st.session_state["menu_filequery_uploaded_file_name"]
    st.caption(f":blue[Session:] {menu_filequery_session_id} | :blue[File:] {menu_filequery_uploaded_file_name}")
elif "menu_filequery_uploaded_file_name" in st.session_state:
    menu_filequery_uploaded_file_name = st.session_state["menu_filequery_uploaded_file_name"]
    st.caption(f":blue[Session:] new | :blue[File:] {menu_filequery_uploaded_file_name}")

####
# File attachment via chat_input (accept_file). The Bedrock EXTERNAL_SOURCES
# call needs the file bytes on EVERY question, but chat_input only delivers
# the file on the submission it was attached to — so the current file is
# persisted in session state and reused for follow-up questions.

submission = st.chat_input(
    "Ask a question — attach a file (max 20000 tokens)",
    accept_file=True,
    file_type=["pdf", "md", "txt", "docx", "html", "csv", "xls", "xlsx",
               "png", "jpg", "jpeg", "gif", "webp"],
)

if submission:
    if submission.files:
        attached = submission.files[0]
        # Reset the KB session when the file changed
        if "menu_filequery_session_id" in st.session_state:
            del st.session_state["menu_filequery_session_id"]
        st.session_state["menu_filequery_uploaded_file_name"] = attached.name
        st.session_state["menu_filequery_uploaded_file_type"] = attached.type
        st.session_state["menu_filequery_uploaded_file_bytes"] = attached.getvalue()

if submission and not submission.text:
    # File attached without a question — acknowledge and wait for the question
    st.info(
        f"File **{st.session_state.get('menu_filequery_uploaded_file_name', '')}** "
        "attached. Ask a question about it."
    )

if submission and submission.text and "menu_filequery_uploaded_file_bytes" not in st.session_state:
    st.warning("Attach a file first — use the 📎 button in the message box.")

if (
    submission
    and submission.text
    and "menu_filequery_uploaded_file_bytes" in st.session_state
):
    prompt = submission.text
    uploaded_file_name  = st.session_state["menu_filequery_uploaded_file_name"]
    uploaded_file_type  = st.session_state["menu_filequery_uploaded_file_type"]
    uploaded_file_bytes = st.session_state["menu_filequery_uploaded_file_bytes"]

    st.session_state.menu_filequery_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if uploaded_file_type in _IMAGE_FORMATS:
        # Image path: RetrieveAndGenerate rejects images, so query the
        # multimodal model directly via the Converse API. Stateless — the
        # image is re-sent with each question, no KB session involved.
        with st.chat_message("assistant"):
            try:
                response = runtime_client.converse(
                    modelId=opt_model_id,
                    messages=[{
                        "role": "user",
                        "content": [
                            {
                                "image": {
                                    "format": _IMAGE_FORMATS[uploaded_file_type],
                                    "source": {"bytes": uploaded_file_bytes},
                                }
                            },
                            {"text": prompt},
                        ],
                    }],
                    # Claude 4.5 rejects temperature + topP together on Converse
                    inferenceConfig={
                        "temperature": opt_temperature,
                        "maxTokens": opt_max_tokens,
                    },
                )
                answer = response["output"]["message"]["content"][0]["text"]
                st.markdown(answer)
                st.caption(f"Image: {uploaded_file_name}")
                st.session_state.menu_filequery_messages.append(
                    {"role": "assistant", "content": answer}
                )
            except ClientError as err:
                message = err.response["Error"]["Message"]
                logger.error("A client error occurred: %s", message)
                st.markdown(f":red[{message}]")
        st.stop()

    with st.chat_message("assistant"):

        try:

            inference_config = {
                "textInferenceConfig": {
                    "temperature": opt_temperature,
                    "maxTokens": opt_max_tokens,
                    "topP": opt_top_p,
                }
            }

            additional_model_fields = {"top_k": opt_top_k}

            params = {
                "input": { "text": prompt },
                "retrieveAndGenerateConfiguration": {
                    "type": "EXTERNAL_SOURCES",
                    "externalSourcesConfiguration": {
                        "modelArn": opt_model_id,
                        "generationConfiguration": {
                            "inferenceConfig": inference_config,
                            "additionalModelRequestFields": {
                                "top_k": opt_top_k
                            },
                            #"guardrailConfiguration": {
                            #    "guardrailId": "",
                            #    "guardrailVersion": ""
                            #},
                        },
                        #"promptTemplate": {
                        #    "textPromptTemplate": ""
                        #},
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
            #st.caption(f"SessionId: {response['sessionId']} | File: {st.session_state['menu_filequery_uploaded_file_name']}")
            st.caption(f"File: {st.session_state['menu_filequery_uploaded_file_name']}")
            with st.expander("citations"):
                st.json(response["citations"])
            
            ###
            #with st.expander("citations"):
            response_citation_id = 1
            response_citations = response["citations"]
            for response_citation in response_citations:
                citation_text = response_citation["generatedResponsePart"]["textResponsePart"]["text"]
                st.caption(f":green[[{response_citation_id}] {citation_text}]")
                citation_references = response_citation["retrievedReferences"]
                for citation_reference in citation_references:
                    citation_reference_text = citation_reference["content"]["text"]
                    #citation_reference_text = citation_reference_text[0:12] + " orange:[" + citation_reference_text[13:25] + "] " + citation_reference_text[26:0]
                    st.caption(f":orange[{citation_reference_text}]")
                response_citation_id = response_citation_id + 1
            ###

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