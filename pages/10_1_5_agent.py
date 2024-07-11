import streamlit as st
import boto3
import cmn_settings
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
import app_bedrock_lib


from botocore.exceptions import BotoCoreError, ClientError

AWS_REGION = cmn_settings.AWS_REGION
MAX_MESSAGES = 100 * 2

AGENT_ID=os.environ.get("AGENT_ID") 
AGENT_ALIAS_ID=os.environ.get("AGENT_ALIAS_ID") 

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
bedrock_agent_runtime = boto3.client('bedrock-agent-runtime', region_name=AWS_REGION)

####################################################################################

st.set_page_config(
    page_title="Agent",
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

opt_agent_list = app_bedrock_lib.list_agents()


# https://docs.aws.amazon.com/bedrock/latest/userguide/model-ids.html
opt_model_id_list = [
    "anthropic.claude-3-5-sonnet-20240620-v1:0",
    "anthropic.claude-3-sonnet-20240229-v1:0",
    "anthropic.claude-3-haiku-20240307-v1:0",
    #"anthropic.claude-3-opus-20240229-v1:0",
    "cohere.command-r-v1:0", # The model returned the following errors: Malformed input request: #: extraneous key [top_k] is not permitted, please reformat your input and try again.
    "cohere.command-r-plus-v1:0",
    "meta.llama2-13b-chat-v1", # Llama 2 Chat 13B
    "meta.llama2-70b-chat-v1", # Llama 2 Chat 70B
    "meta.llama3-8b-instruct-v1:0", # Llama 3 8b Instruct
    "meta.llama3-70b-instruct-v1:0",  # Llama 3 70b Instruct
    #"mistral.mistral-7b-instruct-v0:2", # Mistral 7B Instruct Does not support system message
    #"mistral.mixtral-8x7b-instruct-v0:1", # Mixtral 8X7B Instruct Does not support system message
    "mistral.mistral-small-2402-v1:0", # Mistral Small
    "mistral.mistral-large-2402-v1:0", # Mistral Large
]

with st.sidebar:
    opt_agent_id = st.selectbox(label="Agent ID", options=opt_agent_list, index = 0, key="agent_id")
    opt_model_id = st.selectbox(label="Model ID", options=opt_model_id_list, index = 0, key="model_id")
    opt_temperature = st.slider(label="Temperature", min_value=0.0, max_value=1.0, value=0.1, step=0.1, key="temperature")
    opt_top_p = st.slider(label="Top P", min_value=0.0, max_value=1.0, value=1.0, step=0.1, key="top_p")
    opt_top_k = st.slider(label="Top K", min_value=0, max_value=500, value=250, step=1, key="top_k")
    opt_max_tokens = st.slider(label="Max Tokens", min_value=0, max_value=4096, value=2048, step=1, key="max_tokens")
    opt_system_msg = st.text_area(label="System Message", value="You are a question and answering chatbot", key="system_msg")



st.markdown("💬 Agent")


if "menu_agent_session_id" not in st.session_state:
    session_id = str(uuid.uuid4())
    st.session_state["menu_agent_session_id"] = session_id

st.markdown(st.session_state["menu_agent_session_id"])

if "menu_agent_messages" not in st.session_state:
    st.session_state["menu_agent_messages"] = []

#if "audio_stream" not in st.session_state:
#    st.session_state["audio_stream"] = ""

st.markdown(f"{len(st.session_state.menu_agent_messages)}/{MAX_MESSAGES}")

idx = 1
for msg in st.session_state.menu_agent_messages:
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
    
if "menu_agent_uploader_key" not in st.session_state:
    st.session_state.menu_agent_uploader_key = 0

prompt = st.chat_input()

if prompt:
    
    message_history = st.session_state.menu_agent_messages.copy()
    message_user_latest = {"role": "user", "content": [{ "text": prompt }]}
    
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
    if opt_model_id.startswith("meta"):
        additional_model_fields = None
    if opt_model_id.startswith("mistral"):
        additional_model_fields = None



    #print(json.dumps(inference_config, indent=3))
    #print(json.dumps(system_prompts, indent=3))

    with st.spinner('Processing...'):

        try:
            
            #response = bedrock_runtime.converse_stream(
            #    modelId=opt_model_id,
            #    messages=message_history,
            #    system=system_prompts,
            #    inferenceConfig=inference_config,
            #    additionalModelRequestFields=additional_model_fields
            #)

            r_agent_id = AGENT_ID #opt_agent_id.split(" ")[0]
            r_agent_alias_id = AGENT_ALIAS_ID #opt_agent_id.split(" ")[1]
            print(f"--------{r_agent_id} {r_agent_alias_id}")
            # https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/bedrock-agent-runtime/client/invoke_agent.html
            response = bedrock_agent_runtime.invoke_agent(
                agentId=r_agent_id, 
                agentAliasId=r_agent_alias_id, # Use TSTALIASID as the agentAliasId to invoke the draft version of your agent.
                sessionId=st.session_state["menu_agent_session_id"],  # you continue an existing session with the agent if the value you set for the idle session timeout hasn't been exceeded.
                inputText=prompt, 
                enableTrace=True, 
                endSession=False  # true to end the session with the agent.
            )
            print(f"Answer: {response}")

            result_text = ""
            with st.chat_message("assistant"):
                result_container = st.container(border=True)
                result_area = st.empty()

                event_sequence_id = 0
                answer = ""
                sources = []
                sources_text = ""
                generated_text = ""
                event_stream = response['completion']
                for event in event_stream:        
                    print(f"type={type(event)} event={event}")
                    event_sequence_id += 1
                    if 'chunk' in event:
                        chunk = event['chunk']
                        msg_chunk = ""
                        if 'bytes' in chunk:
                            token = chunk['bytes'].decode("utf-8")
                            answer += token
                            msg_chunk += f"\n \n {token} \n"
                            
                        if 'attribution' in chunk: # If a knowledge base was queried, an attribution object with a list of citations is returned.
                            attribution = chunk['attribution']
                            location_uri = ""
                            for citation in attribution['citations']:
                                # generatedResponsePart object contains the text generated by the model based on the information from the text in the retrievedReferences
                                generated_text = citation['generatedResponsePart']['textResponsePart']['text']
                                # retrievedReferences object contains the exact text in the chunk relevant to the query alongside the S3 location of the data source
                                for reference in citation['retrievedReferences']: 
                                    reference_text = reference['content']['text']
                                    sources_text += reference_text
                                    location = reference['location']
                                    if location['type'] == 's3':
                                        location_uri = location['s3Location']['uri']
                                        sources.append(location_uri)
                            msg_chunk += f"Attribution: Location={location_uri}"

                        #await msg.stream_token(msg_chunk)
                        #await msg.send()
                    elif 'trace' in event:
                        trace = event['trace']['trace']
                        msg_trace = f"{event_sequence_id}. "
                        if 'orchestrationTrace' in trace:
                            msg_trace += f"ORCHESTRATION | "
                            orchestration_trace = trace['orchestrationTrace']
                            print(f"Orchestration Step: \n")
                            if 'rationale' in orchestration_trace:
                                rationale_text = orchestration_trace['rationale']['text']
                                print(f"  Rationale: {rationale_text} \n")
                                msg_trace += f"Rationale: {rationale_text} \n"
                            if 'invocationInput' in orchestration_trace:
                                invocation_input = orchestration_trace['invocationInput']
                                invocation_type = invocation_input['invocationType']
                                invocation_input_kb_id = ""
                                invocation_input_kb_text = ""
                                if 'knowledgeBaseLookupInput' in invocation_input:
                                    invocation_input_kb_id = invocation_input['knowledgeBaseLookupInput']['knowledgeBaseId']
                                    invocation_input_kb_text = invocation_input['knowledgeBaseLookupInput']['text']
                                print(f"  InvocationInput: {invocation_type} KB.ID={invocation_input_kb_id} KB.Text={invocation_input_kb_text} \n")
                                msg_trace += f"InvocationInput: {invocation_type} KB.ID={invocation_input_kb_id} KB.Text={invocation_input_kb_text} \n"
                            if 'observation' in orchestration_trace:
                                observation = orchestration_trace['observation']
                                observation_type = observation['type']
                                kb_type = ""
                                kb_location = ""
                                if 'knowledgeBaseLookupOutput' in observation:
                                    kb_lookup = observation['knowledgeBaseLookupOutput']
                                    for kb in kb_lookup['retrievedReferences']:
                                        kb_type = kb['location']['type']
                                        if 'S3' == kb_type:
                                            kb_location = kb['location']['s3Location']['uri']
                                observation_type = observation['type']
                                print(f"  Observation: {observation_type} KB.Type={kb_type} KB.URI={kb_location} \n")
                                msg_trace += f"Observation: {observation_type} KB.Type={kb_type} KB.URI={kb_location} \n"
                            if 'modelInvocationInput' in orchestration_trace:
                                model_invocation_input = orchestration_trace['modelInvocationInput']['type']
                                print(f"  ModelInvocationInput: {model_invocation_input} \n")
                                msg_trace += f"ModelInvocationInput: {model_invocation_input} \n"

                            #async with cl.Step(name="Orchestration Step") as orchestration_step:
                            #    orchestration_step.input = "Orchestration Step Input"
                            #    orchestration_step.output = msg_trace
                            result_text += f"\n\n{msg_trace}"
                            result_area.write(result_text)

                        elif 'preProcessingTrace' in trace:
                            print(f"PreProcessing Step: \n")
                            msg_trace += f"PRE_PROCESSING | "
                            pre_processing_trace = trace['preProcessingTrace']
                            if 'modelInvocationInput' in pre_processing_trace:
                                model_invocation_input_type = pre_processing_trace['modelInvocationInput']['type']
                                model_invocation_input_text = pre_processing_trace['modelInvocationInput']['text']
                                #print(f"  ModelInvocationInput: Type={model_invocation_input_type} Text={model_invocation_input_text} \n")
                                print(f"  ModelInvocationInput: Type={model_invocation_input_type} \n")
                                msg_trace += f"ModelInvocationInput: Type={model_invocation_input_type} \n"
                            if 'modelInvocationOutput' in pre_processing_trace:
                                model_invocation_output = pre_processing_trace['modelInvocationOutput']
                                model_invocation_output_rationale = model_invocation_output['parsedResponse']['rationale']
                                model_invocation_output_isvalid = model_invocation_output['parsedResponse']['isValid']
                                print(f"  ModelInvocationOutput: Rationale={model_invocation_output_rationale} IsValid={model_invocation_output_isvalid} \n")
                                msg_trace += f"ModelInvocationOutput: Rationale={model_invocation_output_rationale} IsValid={model_invocation_output_isvalid} \n"
                            
                            #await step.stream_token(msg_trace)
                            #async with cl.Step(name="Pre Processing Step", show_input=False) as pre_processing_step:
                            #    pre_processing_step.input = "Pre Processing Step Input"
                            #    pre_processing_step.output = msg_trace
                            result_text += f"\n\n{msg_trace}"
                            result_area.write(result_text)

                        elif 'postProcessingTrace' in trace:
                            msg_trace += f"POST_PROCESSIG |  \n"

                            #async with cl.Step(name="Post Processing Step") as post_processing_step:
                            #    post_processing_step.input = "Post Processing Step Input"
                            #    post_processing_step.output = msg_trace

                        elif 'failureTrace' in trace:
                            msg_trace += f"FAILURE |  \n"

                            failure_reason = trace["failureTrace"]["failureReason"]

                            msg_trace += f"{failure_reason}"

                            #async with cl.Step(name="Failure Step") as failure_step:
                            #    failure_step.input = "Failure Step Input"
                            #    failure_step.output = msg_trace

                        else:
                            msg_trace += f"UNKNOWN | \n"

                            #async with cl.Step(name="Unknown Step") as unknown_step:
                            #    unknown_step.input = "Unkwnown Step Input"
                            #    unknown_step.output = msg_trace

                        #await msg.stream_token(msg_trace)    
                        #await msg.send()
                    else:
                        #await msg.stream_token(".")
                        #await msg.send()
                        pass
                print("*********************************************************")
                print(f"Answer: {answer}")
                print("*********************************************************")
                print(f"Sources: {sources}")
                print("*********************************************************")
                print(f"Sources Text: {sources_text}")
                print("*********************************************************")
                print(f"Generated Text: {generated_text}")
                print("*********************************************************")

            result_text += f"\n\n{answer}"
            result_area.write(result_text)
            
            message_assistant_latest = {"role": "assistant", "content": [{ "text": result_text }]}


            st.session_state.menu_agent_messages.append(message_user_latest)
            st.session_state.menu_agent_messages.append(message_assistant_latest)

            
            # Trim message History
            menu_agent_messages = st.session_state.menu_agent_messages
            menu_agent_messages_len = len(menu_agent_messages)
            if menu_agent_messages_len > MAX_MESSAGES:
                del menu_agent_messages[0 : (menu_agent_messages_len - MAX_MESSAGES) * 2] #make sure we remove both the user and assistant responses
            #print(f"menu_agent_messages_len={menu_agent_messages_len}")


            #print(json.dumps(message_user_latest, indent=2))
            #print(message_user_latest)

        except ClientError as err:
            message = err.response["Error"]["Message"]
            logger.error("A client error occurred: %s", message)
            print("A client error occured: " + format(message))
            st.chat_message("system").write(message)

if "audio_stream" in st.session_state and st.session_state["audio_stream"] != "":
    audio_bytes = BytesIO(st.session_state['audio_stream'])
    st.audio(audio_bytes, format='audio/mp3', autoplay=False)