import boto3
import streamlit as st
import cmn_auth
import cmn_settings
import logging
from botocore.exceptions import ClientError
import app_bedrock_lib

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
bedrock = boto3.client("bedrock", region_name=AWS_REGION)
bedrock_runtime = boto3.client('bedrock-runtime', region_name=AWS_REGION)
bedrock_agent_runtime = boto3.client("bedrock-agent-runtime", region_name=AWS_REGION)


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

#opt_model_id_list = app_bedrock_lib.bedrock_list_models(bedrock)
#print(opt_model_id_list)

opt_mode_id_list = [
    "mark_1",
    "mark_2",
]

opt_model_id_list = [
    "anthropic.claude-3-sonnet-20240229-v1:0",
    #"anthropic.claude-3-5-sonnet-20241022-v2:0", #The provided model is not supported for EXTERNAL_SOURCES RetrieveAndGenerateType. Update the model arn then retry your request.
    #"anthropic.claude-3-5-sonnet-20240620-v1:0", #The provided model is not supported for EXTERNAL_SOURCES RetrieveAndGenerateType. Update the model arn then retry your request.
    "anthropic.claude-3-haiku-20240307-v1:0"
]

opt_converse_model_id_list = [
    "us.anthropic.claude-3-5-sonnet-20240620-v1:0",
    "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
]

opt_top_k = 250
opt_top_p = 1.0

with st.sidebar:
    #st.markdown(":blue[Settings]")
    opt_mode_id = st.selectbox(label="Mode ID", options=opt_mode_id_list, index = 0, key="mode_id")


with st.sidebar:
    if opt_mode_id == "mark_1":
        opt_model_id = st.selectbox(label="Model ID", options=opt_model_id_list, index = 0, key="model_id")
        opt_temperature = st.slider(label="Temperature", min_value=0.0, max_value=1.0, value=0.0, step=0.1, key="temperature")
        #opt_top_p = st.slider(label="Top P", min_value=0.0, max_value=1.0, value=1.0, step=0.1, key="top_p")
        #opt_top_k = st.slider(label="Top K", min_value=0, max_value=500, value=250, step=1, key="top_k")
        opt_max_tokens = st.slider(label="Max Tokens", min_value=0, max_value=4096, value=2048, step=1, key="max_tokens")
    else:
        #st.markdown(":blue[Settings]")
        opt_model_id = st.selectbox(label="Model ID", options=opt_converse_model_id_list, index = 0, key="model_id")
        opt_temperature = st.slider(label="Temperature", min_value=0.0, max_value=1.0, value=0.0, step=0.1, key="temperature")
        #opt_top_p = st.slider(label="Top P", min_value=0.0, max_value=1.0, value=1.0, step=0.1, key="top_p")
        #opt_top_k = st.slider(label="Top K", min_value=0, max_value=500, value=250, step=1, key="top_k")
        opt_max_tokens = st.slider(label="Max Tokens", min_value=0, max_value=4096, value=2048, step=1, key="max_tokens")



st.title("File Query v2.4")
st.write("Knowledge Bases on File Upload for Amazon Bedrock")

if "menu_docquery_messages" not in st.session_state:
    st.session_state.menu_docquery_messages = []

for message in st.session_state.menu_docquery_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "citations" in message:
            with st.expander("citations"):
                st.json(message["citations"])


if "menu_docquery_session_id" in st.session_state:
    menu_docquery_session_id = st.session_state["menu_docquery_session_id"]
    menu_docquery_uploaded_file_name = st.session_state["menu_docquery_uploaded_file_name"]
    st.caption(f":blue[Session:] {menu_docquery_session_id} | :blue[File:] {menu_docquery_uploaded_file_name}")
elif "menu_docquery_uploaded_file_name" in st.session_state:    
    menu_docquery_uploaded_file_name = st.session_state["menu_docquery_uploaded_file_name"]
    st.caption(f":blue[Session:] new | :blue[File:] {menu_docquery_uploaded_file_name}")

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
    uploaded_file_key = uploaded_file.name.replace(".", "_").replace(" ", "_")
    # Reset the session key when the file changed
    if "menu_docquery_session_id" in st.session_state:
        del st.session_state["menu_docquery_session_id"]
    st.session_state["menu_docquery_uploaded_file_name"] = uploaded_file_name

#### 

if prompt := st.chat_input(
    "Ask question about the file." if uploaded_file else "Choose and upload a file first",
    disabled=(not uploaded_file),
):
    
    with st.chat_message("user"):
        st.markdown(prompt)

    
    with st.chat_message("assistant"):
        
        try:

            if opt_mode_id == "mark_1":

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

                if "menu_docquery_session_id" in st.session_state:
                    params["sessionId"] = st.session_state["menu_docquery_session_id"]

                response = bedrock_agent_runtime.retrieve_and_generate(**params)

                
                st.markdown(response["output"]["text"])
                #st.caption(f"SessionId: {response['sessionId']} | File: {st.session_state['menu_docquery_uploaded_file_name']}")
                st.caption(f"File: {st.session_state['menu_docquery_uploaded_file_name']}")
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

                st.session_state["menu_docquery_session_id"] = response["sessionId"]      
                st.session_state.menu_docquery_messages.append({"role": "user", "content": prompt})          
                st.session_state.menu_docquery_messages.append(
                    {
                        "role": "assistant",
                        "content": response["output"]["text"],
                        "citations": response["citations"],
                    }
                )

            else:

                message_history = st.session_state.menu_docquery_messages.copy()

                message_user_latest = {"role": "user", "content": [{ "text": prompt }]}
                if uploaded_file_name:
                    content = message_user_latest['content']
                    #'pdf'|'csv'|'doc'|'docx'|'xls'|'xlsx'|'html'|'txt'|'md',
                    mime_mapping_document = {
                        "text/plain": "txt",
                        "application/vnd.ms-excel": "csv",
                        "application/pdf": "pdf",
                    }
                    if uploaded_file.type in mime_mapping_document:
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

                system_prompts = [{"text" : "You are a helpful assistant"}]

                additional_model_fields = {}
                
                inference_config = {
                    "temperature": opt_temperature,
                    "maxTokens": opt_max_tokens,
                    "topP": opt_top_p,
                    #stopSequences 
                }
                response = bedrock_runtime.converse_stream(
                    modelId=opt_model_id,
                    messages=message_history,
                    system=system_prompts,
                    inferenceConfig=inference_config,
                    additionalModelRequestFields=additional_model_fields
                )
        

                #with st.chat_message("assistant", avatar=setAvatar("assistant")):
                result_text = ""
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

                    #st.session_state.menu_docquery_messages.append({"role": "user", "content": [{ "text": prompt }]})          
                    #st.session_state.menu_docquery_messages.append(
                    #    {
                    #        "role": "assistant",
                    #        "content": [{ "text": result_text }],
                    #        #"citations": response["citations"],
                    #    }
                    #)

        except ClientError as err:
            message = err.response["Error"]["Message"]
            logger.error("A client error occurred: %s", message)
            st.markdown(f":red[{message}]")
            #st.chat_message("system").write(message)