import streamlit as st
import boto3
import cmn_settings
import json
import logging
import cmn_auth
import os
from PIL import Image
import io
import base64
import uuid
import pandas as pd
from cmn.bedrock_converse_tools import CalculatorBedrockConverseTool
from cmn.bedrock_converse_tools_2 import AcronymBedrockConverseTool
from cmn.bedrock_converse_tools_url import UrlContentBedrockConverseTool
from cmn.bedrock_converse_tools_wikipedia import WikipediaBedrockConverseTool

from botocore.exceptions import BotoCoreError, ClientError

AWS_REGION = cmn_settings.AWS_REGION
MAX_MESSAGES = 100 * 2

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

settings = {
    "enable_print_invocation_metrics": True
}

####################################################################################

bedrock_runtime = boto3.client('bedrock-runtime', region_name=AWS_REGION)
calculator_tool = CalculatorBedrockConverseTool()
acronym_tool = AcronymBedrockConverseTool()
url_loader_tool = UrlContentBedrockConverseTool()
wikipedia_tool = WikipediaBedrockConverseTool()

tools = [calculator_tool, acronym_tool, url_loader_tool, wikipedia_tool]

tool_config = {
        "tools": [tool.definition for tool in tools]
    }

####################################################################################

def image_to_base64(image, mime_type:str):
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

mime_mapping_document = {
    "text/plain": "txt",
    "application/vnd.ms-excel": "csv",
    "application/pdf": "pdf",
}

def invoke_llm(message_history, system_prompts, tool_config, inference_config, additional_model_fields):
    try:
        tool_invocation = {
            "tool_name": None
        }
        tool_input = ""

        response = bedrock_runtime.converse_stream(
            modelId=opt_model_id,
            messages=message_history,
            system=system_prompts,
            toolConfig=tool_config,
            inferenceConfig=inference_config,
            additionalModelRequestFields=additional_model_fields
        )

        result_text = ""
        with st.chat_message("assistant"):
            result_container = st.container(border=True)
            result_area = st.empty()
            stream = response.get('stream')
            for event in stream:

                if 'messageStart' in event:
                    print('messageStart')
                    pass

                if 'contentBlockStart' in event:
                    content_block_start = event['contentBlockStart']
                    print(content_block_start)
                    if 'start' in content_block_start:
                        content_block_start_start = content_block_start['start']
                        if 'toolUse' in content_block_start_start:
                            content_block_tool_use = content_block_start_start['toolUse']
                            tool_use_id = content_block_tool_use['toolUseId']
                            tool_use_name = content_block_tool_use['name']
                            print(f"tool_use_id={tool_use_id} tool_use_name={tool_use_name}")
                            tool_invocation['tool_name'] = tool_use_name
                            tool_invocation['tool_use_id'] = tool_use_id

                if 'contentBlockDelta' in event:
                    content_delta = event['contentBlockDelta']['delta']
                    print(f"content_delta {content_delta}")
                    if 'text' in content_delta:
                        result_text += f"{content_delta['text']}"
                        result_area.write(result_text)
                    if 'toolUse' in content_delta:
                        content_delta_tool_input = content_delta['toolUse']['input']
                        tool_input += content_delta_tool_input

                if 'messageStop' in event:
                    print(f"messageStop")
                    stop_reason = event['messageStop']['stopReason']
                    if stop_reason == 'end_turn':
                        pass
                    elif "tool_use" == stop_reason:
                        tool_input_json = json.loads(tool_input)
                        print(tool_input_json)
                        tool_invocation['tool_arguments'] = tool_input
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

                    if settings["enable_print_invocation_metrics"]:
                        stats = f"M| token.in={input_token_count} token.out={output_token_count} token={total_token_count} latency={latency}"
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

    except ClientError as err:
        message = err.response["Error"]["Message"]
        logger.error("A client error occurred: %s", message)
        print("A client error occured: " + format(message))
        st.chat_message("system").write(message)


opt_model_id_list = [
    "global.anthropic.claude-sonnet-4-5-20250929-v1:0",
    "global.anthropic.claude-sonnet-4-20250514-v1:0",
    #"us.anthropic.claude-3-7-sonnet-20250219-v1:0",
    "cohere.command-r-v1:0",
    "cohere.command-r-plus-v1:0",
    "meta.llama2-13b-chat-v1",
    "meta.llama2-70b-chat-v1",
    "meta.llama3-8b-instruct-v1:0",
    "meta.llama3-70b-instruct-v1:0",
    "mistral.mistral-small-2402-v1:0",
    "mistral.mistral-large-2402-v1:0",
]

opt_top_p = 1.0
opt_top_k = 250
with st.sidebar:
    opt_model_id = st.selectbox(label="Model ID", options=opt_model_id_list, index=0, key="model_id")
    opt_temperature = st.slider(label="Temperature", min_value=0.0, max_value=1.0, value=0.1, step=0.1, key="temperature")
    #opt_top_p = st.slider(label="Top P", min_value=0.0, max_value=1.0, value=1.0, step=0.1, key="top_p")
    #opt_top_k = st.slider(label="Top K", min_value=0, max_value=500, value=250, step=1, key="top_k")
    opt_max_tokens = st.slider(label="Max Tokens", min_value=0, max_value=4096, value=2048, step=1, key="max_tokens")
    opt_system_msg = st.text_area(label="System Message", value="You are a question and answering chatbot", key="system_msg")

st.markdown("💬 Converse Tool 375")

if "menu_converse_tool_messages" not in st.session_state:
    st.session_state["menu_converse_tool_messages"] = []

st.markdown(f"{len(st.session_state.menu_converse_tool_messages)}/{MAX_MESSAGES}")

idx = 1
for msg in st.session_state.menu_converse_tool_messages:
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

if "menu_converse_tool_uploader_key" not in st.session_state:
    st.session_state.menu_converse_tool_uploader_key = 0

uploaded_file = st.file_uploader(
        "Attach Image",
        type=["PNG", "JPEG", "TXT", "CSV", "PDF", "MD"],
        accept_multiple_files=False,
        label_visibility="collapsed",
        key=f"menu_converse_tool_uploader_key_{st.session_state.menu_converse_tool_uploader_key}"
    )

tool_names = ", ".join([tool.definition['toolSpec']['name'] for tool in [calculator_tool, acronym_tool, url_loader_tool]])
st.markdown(f"Tools: {tool_names}")

prompt = st.chat_input()

uploaded_file_key = None
uploaded_file_name = None
uploaded_file_bytes = None
uploaded_file_type = None
uploaded_file_base64 = None
if uploaded_file:
    if uploaded_file.type in mime_mapping_image:
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
                uploaded_file_df = pd.read_csv(uploaded_file, encoding="utf-8")
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
        print(f"******{uploaded_file.type}")

if prompt:
    message_history = st.session_state.menu_converse_tool_messages.copy()
    message_user_latest = {"role": "user", "content": [{"text": prompt}]}
    if uploaded_file_name:
        content = message_user_latest['content']
        if uploaded_file_type in mime_mapping_image:
            content.append(
                {
                    "image": {
                        "format": mime_mapping_image[uploaded_file_type],
                        "source": {
                            "bytes": uploaded_file_bytes,
                        }
                    },
                }
            )
        elif uploaded_file.type in mime_mapping_document:
            uploaded_file_name_clean = uploaded_file_key
            content.append(
                {
                    "document": {
                        "format": mime_mapping_document[uploaded_file_type],
                        "name": uploaded_file_name_clean,
                        "source": {
                            "bytes": uploaded_file_bytes,
                        }
                    },
                }
            )
    message_history.append(message_user_latest)
    st.chat_message("user").write(prompt)

    system_prompts = [{"text": opt_system_msg}]

    inference_config = {
        "temperature": opt_temperature,
        "maxTokens": opt_max_tokens,
    }

    additional_model_fields = None

    with st.spinner('Processing...'):
        try:
            tool_invocation = {
                "tool_name": None
            }
            tool_input = ""

            response = bedrock_runtime.converse_stream(
                modelId=opt_model_id,
                messages=message_history,
                system=system_prompts,
                toolConfig=tool_config,
                inferenceConfig=inference_config,
                additionalModelRequestFields=additional_model_fields
            )

            result_text = ""
            with st.chat_message("assistant"):
                result_container = st.container(border=True)
                result_area = st.empty()
                stream = response.get('stream')
                for event in stream:

                    if 'messageStart' in event:
                        pass

                    if 'contentBlockStart' in event:
                        content_block_start = event['contentBlockStart']
                        print(content_block_start)
                        if 'start' in content_block_start:
                            content_block_start_start = content_block_start['start']
                            if 'toolUse' in content_block_start_start:
                                content_block_tool_use = content_block_start_start['toolUse']
                                tool_use_id = content_block_tool_use['toolUseId']
                                tool_use_name = content_block_tool_use['name']
                                print(f"tool_use_id={tool_use_id} tool_use_name={tool_use_name}")
                                tool_invocation['tool_name'] = tool_use_name
                                tool_invocation['tool_use_id'] = tool_use_id

                    if 'contentBlockDelta' in event:
                        content_delta = event['contentBlockDelta']['delta']
                        if 'text' in content_delta:
                            result_text += f"{content_delta['text']}"
                            result_area.write(result_text)
                        if 'toolUse' in content_delta:
                            content_delta_tool_input = content_delta['toolUse']['input']
                            tool_input += content_delta_tool_input

                    if 'messageStop' in event:
                        stop_reason = event['messageStop']['stopReason']
                        if stop_reason == 'end_turn':
                            pass
                        elif "tool_use" == stop_reason:
                            print(f"input: {tool_input}")
                            tool_input_json = json.loads(tool_input)
                            print(tool_input_json)
                            tool_invocation['tool_arguments'] = tool_input
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
                        if settings["enable_print_invocation_metrics"]:
                            stats = f"L| token.in={input_token_count} token.out={output_token_count} token={total_token_count} latency={latency}"
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

            print(tool_invocation)

            if tool_invocation['tool_name'] != None:
                tool_name = tool_invocation['tool_name']
                tool_args_json = json.loads(tool_invocation['tool_arguments'])

                tool_request_message = {
                    "role": "assistant",
                    "content": [
                        {
                            "toolUse": {
                                "toolUseId": tool_invocation['tool_use_id'],
                                "name": tool_invocation['tool_name'],
                                "input": json.loads(tool_invocation['tool_arguments'])
                            }
                        }
                    ]
                }

                if calculator_tool.matches(tool_name):
                    expr_result = calculator_tool.invoke(tool_args_json['expression'])
                    tool_result_message = {
                        "role": "user",
                        "content": [
                            {
                                "toolResult": {
                                        "toolUseId": tool_invocation['tool_use_id'],
                                        "content": [{"json": {"expr_result": expr_result}}]
                                    }
                            }
                        ]
                    }

                if acronym_tool.matches(tool_name):
                    expr_result = acronym_tool.invoke(tool_args_json['expression'])
                    tool_result_message = {
                        "role": "user",
                        "content": [
                            {
                                "toolResult": {
                                        "toolUseId": tool_invocation['tool_use_id'],
                                        "content": [{"json": {"expr_result": expr_result}}]
                                    }
                            }
                        ]
                    }

                if url_loader_tool.matches(tool_name):
                    expr_result = url_loader_tool.invoke(tool_args_json['expression'])
                    tool_result_message = {
                        "role": "user",
                        "content": [
                            {
                                "toolResult": {
                                        "toolUseId": tool_invocation['tool_use_id'],
                                        "content": [{"json": {"expr_result": expr_result}}]
                                    }
                            }
                        ]
                    }

                for tool in tools:
                    if tool.matches(tool_name):
                        expr_result = tool.invoke(tool_args_json['expression'])
                        tool_result_message = {
                            "role": "user",
                            "content": [
                                {
                                    "toolResult": {
                                            "toolUseId": tool_invocation['tool_use_id'],
                                            "content": [{"json": {"expr_result": expr_result}}]
                                        }
                                }
                            ]
                        }
                        break

                messages = [message_user_latest, tool_request_message, tool_result_message]

                result_text += f"\n\n:blue[Tool Request: {json.dumps(tool_request_message, indent=2)}]\n"
                result_area.markdown(result_text)

                result_text += f"\n\nTool Result: {json.dumps(tool_result_message, indent=2)}\n"
                result_area.markdown(result_text)

                invoke_llm(messages, system_prompts, tool_config, inference_config, additional_model_fields)

            message_assistant_latest = {"role": "assistant", "content": [{"text": result_text}]}

            st.session_state.menu_converse_tool_messages.append(message_user_latest)
            st.session_state.menu_converse_tool_messages.append(message_assistant_latest)

            # Trim message History
            menu_converse_tool_messages = st.session_state.menu_converse_tool_messages
            menu_converse_tool_messages_len = len(menu_converse_tool_messages)
            if menu_converse_tool_messages_len > MAX_MESSAGES:
                del menu_converse_tool_messages[0 : (menu_converse_tool_messages_len - MAX_MESSAGES) * 2]

            if uploaded_file_name:
                st.session_state.menu_converse_tool_uploader_key += 1

        except ClientError as err:
            message = err.response["Error"]["Message"]
            logger.error("A client error occurred: %s", message)
            print("A client error occured: " + format(message))
            st.chat_message("system").write(message)