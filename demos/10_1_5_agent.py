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

bedrock_runtime = boto3.client('bedrock-runtime', region_name=AWS_REGION)
bedrock_agent_runtime = boto3.client('bedrock-agent-runtime', region_name=AWS_REGION)

st.set_page_config(
    page_title="Agent",
    page_icon="🧊",
    layout="centered",
    initial_sidebar_state="expanded",
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

def image_to_base64(image, mime_type:str):
    buffer = io.BytesIO()
    image.save(buffer, format=mime_type)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

mime_mapping_image = {
    "image/png": "png",
    "image/jpeg": "jpeg",
    "image/jpg": "jpeg",
    "image/gif": "gif",
    "image/webp": "webp",
}

mime_mapping_document = {
    "text/plain": "txt",
    "application/vnd.ms-excel": "csv",
    "application/pdf": "pdf",
}

opt_agent_list = app_bedrock_lib.list_agents()

with st.sidebar:
    opt_agent_id = st.selectbox(label="Agent ID", options=opt_agent_list, index=0, key="agent_id")
    opt_system_msg = st.text_area(label="System Message", value="You are a question and answering chatbot", key="system_msg")

st.markdown("💬 Agent")

if "menu_agent_session_id" not in st.session_state:
    session_id = str(uuid.uuid4())
    st.session_state["menu_agent_session_id"] = session_id

st.markdown(st.session_state["menu_agent_session_id"])

if "menu_agent_messages" not in st.session_state:
    st.session_state["menu_agent_messages"] = []

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
            st.markdown(f"{content_text}")

if "menu_agent_uploader_key" not in st.session_state:
    st.session_state.menu_agent_uploader_key = 0

prompt = st.chat_input()

if prompt:

    message_history = st.session_state.menu_agent_messages.copy()
    message_user_latest = {"role": "user", "content": [{"text": prompt}]}

    message_history.append(message_user_latest)
    st.chat_message("user").write(prompt)

    with st.spinner('Processing...'):

        try:
            r_agent_id = AGENT_ID
            r_agent_alias_id = AGENT_ALIAS_ID
            print(f"--------{r_agent_id} {r_agent_alias_id}")

            response = bedrock_agent_runtime.invoke_agent(
                agentId=r_agent_id,
                agentAliasId=r_agent_alias_id,
                sessionId=st.session_state["menu_agent_session_id"],
                inputText=prompt,
                enableTrace=True,
                endSession=False
            )
            print(f"Response: {response}")

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

                        if 'attribution' in chunk:
                            attribution = chunk['attribution']
                            location_uri = ""
                            for citation in attribution['citations']:
                                generated_text = citation['generatedResponsePart']['textResponsePart']['text']
                                for reference in citation['retrievedReferences']:
                                    reference_text = reference['content']['text']
                                    sources_text += reference_text
                                    location = reference['location']
                                    if location['type'] == 's3':
                                        location_uri = location['s3Location']['uri']
                                        sources.append(location_uri)
                            msg_chunk += f"Attribution: Location={location_uri}"

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

                            result_text += f"\n\n{msg_trace}"
                            result_area.write(result_text)

                        elif 'preProcessingTrace' in trace:
                            print(f"PreProcessing Step: \n")
                            msg_trace += f"PRE_PROCESSING | "
                            pre_processing_trace = trace['preProcessingTrace']

                            if 'modelInvocationInput' in pre_processing_trace:
                                model_invocation_input_type = pre_processing_trace['modelInvocationInput']['type']
                                print(f"  ModelInvocationInput: Type={model_invocation_input_type} \n")
                                msg_trace += f"ModelInvocationInput: Type={model_invocation_input_type} \n"

                            if 'modelInvocationOutput' in pre_processing_trace:
                                model_invocation_output = pre_processing_trace['modelInvocationOutput']
                                model_invocation_output_rationale = model_invocation_output['parsedResponse']['rationale']
                                model_invocation_output_isvalid = model_invocation_output['parsedResponse']['isValid']
                                print(f"  ModelInvocationOutput: Rationale={model_invocation_output_rationale} IsValid={model_invocation_output_isvalid} \n")
                                msg_trace += f"ModelInvocationOutput: Rationale={model_invocation_output_rationale} IsValid={model_invocation_output_isvalid} \n"

                            result_text += f"\n\n{msg_trace}"
                            result_area.write(result_text)

                        elif 'postProcessingTrace' in trace:
                            msg_trace += f"POST_PROCESSING | \n"
                            result_text += f"\n\n{msg_trace}"
                            result_area.write(result_text)

                        elif 'failureTrace' in trace:
                            msg_trace += f"FAILURE | \n"
                            failure_reason = trace["failureTrace"]["failureReason"]
                            msg_trace += f"{failure_reason}"
                            result_text += f"\n\n{msg_trace}"
                            result_area.write(result_text)

                        else:
                            msg_trace += f"UNKNOWN | \n"
                            result_text += f"\n\n{msg_trace}"
                            result_area.write(result_text)

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

            message_assistant_latest = {"role": "assistant", "content": [{"text": result_text}]}

            st.session_state.menu_agent_messages.append(message_user_latest)
            st.session_state.menu_agent_messages.append(message_assistant_latest)

            # Trim message history
            menu_agent_messages = st.session_state.menu_agent_messages
            menu_agent_messages_len = len(menu_agent_messages)
            if menu_agent_messages_len > MAX_MESSAGES:
                del menu_agent_messages[0 : (menu_agent_messages_len - MAX_MESSAGES) * 2]

        except ClientError as err:
            message = err.response["Error"]["Message"]
            logger.error("A client error occurred: %s", message)
            print("A client error occured: " + format(message))
            st.chat_message("system").write(message)