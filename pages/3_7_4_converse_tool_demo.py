"""
Converse Tool Demo - Version 3.7.4 (Prototype)
==============================================

OVERVIEW
--------
This is a prototype implementation of AWS Bedrock Converse API with tool use capabilities.
It demonstrates the basic conversational AI flow with single tool execution support.

STATUS: PROTOTYPE / REFERENCE ONLY
For production use, see version 3.7.5 (3_7_5_converse_tool_v3.py).

ARCHITECTURE
-----------
Monolithic design with all logic inline in a single file. Stream parsing, tool invocation,
and UI rendering are tightly coupled within the same code blocks. This approach works for
simple prototypes and learning but becomes difficult to maintain and test as complexity grows.

KEY CHARACTERISTICS
------------------
- Single tool call per turn: Handles only one tool invocation followed by one LLM follow-up call
- Inline stream parsing: AWS Bedrock event stream parsing logic embedded directly in UI code
- Manual tool matching: Iterates through tool list to find matching tool by name
- Simple metrics: Displays token counts per LLM call without accumulation across turns
- Static system prompt: Tool descriptions must be manually maintained
- No rich rendering: Tool results displayed as JSON dumps only
- Tightly coupled: Business logic mixed with Streamlit UI code, cannot be tested independently

LIMITATIONS
-----------
Cannot handle multi-step tool chains where the model needs to call multiple tools in sequence
or loop back for additional tool calls based on previous results. For example, if the model
needs to fetch data, perform a calculation, and then create a chart, it can only execute the
first tool and must stop. Complex workflows requiring iterative tool use are not supported.

TOOL SUPPORT
-----------
Includes five basic tools: calculator for mathematical operations, acronym for expanding
abbreviations, url_content for fetching web page content, wikipedia for searching Wikipedia,
and datetime for current date/time information. Tools are instantiated as global objects
and collected in a hardcoded list.

FILE STRUCTURE
-------------
All functionality exists in this single file with approximately 481 lines. Stream parsing
happens inline within the prompt handling block. Tool invocation logic is embedded directly
after stream processing. Error handling is scattered throughout with display logic mixed in.

WHEN TO USE
-----------
Use this version for learning how Bedrock Converse API works, creating quick prototypes,
or demonstrating basic tool use concepts. For any production deployment or complex multi-tool
scenarios, migrate to version 3.7.5 which provides proper separation of concerns, full
multi-tool loop support, and extensive rich rendering capabilities.

DOCUMENTATION
------------
See design/converse-design.md for detailed architectural comparison with v3.7.5, including
code examples, flow diagrams, and migration guidance.

Version: 3.7.4
Last Updated: 2026-05-31
"""

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
from cmn.bedrock_converse_tools_acronym import AcronymBedrockConverseTool
from cmn.bedrock_converse_tools_url import UrlContentBedrockConverseTool
from cmn.bedrock_converse_tools_wikipedia import WikipediaBedrockConverseTool
from cmn.bedrock_converse_tools_datetime import DateTimeBedrockConverseTool
from cmn.bedrock_models import FOUNDATION_MODELS, Feature, FoundationModel
from cmn.bedrock.converse.request_builder import InferenceSettings, build_inference_config, build_additional_model_fields

from botocore.exceptions import BotoCoreError, ClientError

AWS_REGION = cmn_settings.AWS_REGION
MAX_MESSAGES = 100 * 2

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

####################################################################################

st.set_page_config(
    page_title="Converse Tool",
    layout="wide", # wide: page hosts chat + info panel side by side
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': None
    },
)

# Main layout with right side Info Panel (invocation metrics, tool use log)
main_col, right_sidebar_col = st.columns([3, 1])

####################################################################################

@st.cache_resource
def get_bedrock_runtime(region=None):
    """bedrock-runtime client for a model's region override (FoundationModel.region), else AWS_REGION"""
    return boto3.client('bedrock-runtime', region_name=region or AWS_REGION)

calculator_tool = CalculatorBedrockConverseTool()
acronym_tool = AcronymBedrockConverseTool()
url_loader_tool = UrlContentBedrockConverseTool()
wikipedia_tool = WikipediaBedrockConverseTool()
datetime_tool = DateTimeBedrockConverseTool()

tools = [calculator_tool, acronym_tool, url_loader_tool, wikipedia_tool, datetime_tool]

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

def read_invocation_metrics(metadata):
    """Token usage and latency from a converse_stream metadata event."""
    usage = metadata.get("usage", {})
    return {
        "input_tokens": usage.get("inputTokens", 0),
        "output_tokens": usage.get("outputTokens", 0),
        "total_tokens": usage.get("totalTokens", 0),
        "latency_ms": metadata.get("metrics", {}).get("latencyMs", 0),
    }

def invoke_llm(message_history, system_prompts, tool_config, inference_config, additional_model_fields, assistant_message):
    """
    Follow-up call after the tool results, streamed into the same assistant bubble.
    Returns (response text, invocation metrics or None, tool uses the model requested).
    This prototype runs one round of tools per turn, so requested tool uses are not run.
    """
    call_metrics = None
    result_text = ""
    tool_uses = {}  # contentBlockIndex -> {"toolUseId", "name", "input" (JSON text)}
    try:
        response = get_bedrock_runtime(opt_fm.region).converse_stream(
            modelId=opt_model_id,
            messages=message_history,
            system=system_prompts,
            toolConfig=tool_config,
            inferenceConfig=inference_config,
            additionalModelRequestFields=additional_model_fields
        )

        with assistant_message:
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
                            print(f"tool_use_id={content_block_tool_use['toolUseId']} tool_use_name={content_block_tool_use['name']}")
                            tool_uses[content_block_start['contentBlockIndex']] = {
                                "toolUseId": content_block_tool_use['toolUseId'],
                                "name": content_block_tool_use['name'],
                                "input": "",
                            }

                if 'contentBlockDelta' in event:
                    content_delta = event['contentBlockDelta']['delta']
                    print(f"content_delta {content_delta}")
                    if 'text' in content_delta:
                        result_text += f"{content_delta['text']}"
                        result_area.write(result_text)
                    if 'toolUse' in content_delta:
                        # Each tool use streams its own input; several can come in one response
                        tool_uses[event['contentBlockDelta']['contentBlockIndex']]["input"] += content_delta['toolUse']['input']

                if 'messageStop' in event:
                    print(f"messageStop")
                    stop_reason = event['messageStop']['stopReason']
                    if stop_reason == 'end_turn':
                        pass
                    elif "tool_use" == stop_reason:
                        pass
                    else:
                        stop_reason_display = stop_reason
                        if stop_reason == 'max_tokens':
                            stop_reason_display = "Insufficient Tokens. Increaes MaxToken Settings."
                        result_text_error = f"{result_text}\n\n:red[Generation Stopped: {stop_reason_display}]"
                        result_area.write(result_text_error)

                if 'metadata' in event:
                    call_metrics = read_invocation_metrics(event['metadata'])

                if "internalServerException" in event:
                    exception = event["internalServerException"]
                    result_text += f"\n\n{exception}"
                    result_area.write(result_text)
                if "modelStreamErrorException" in event:
                    exception = event["modelStreamErrorException"]
                    result_text += f"\n\n{exception}"
                    result_area.write(result_text)
                if "throttlingException" in event:
                    exception = event["throttlingException"]
                    result_text += f"\n\n{exception}"
                    result_area.write(result_text)
                if "validationException" in event:
                    exception = event["validationException"]
                    result_text += f"\n\n{exception}"
                    result_area.write(result_text)

    except ClientError as err:
        message = err.response["Error"]["Message"]
        logger.error("A client error occurred: %s", message)
        print("A client error occured: " + format(message))
        st.chat_message("system").write(message)

    return result_text, call_metrics, list(tool_uses.values())


# Models come from FOUNDATION_MODELS (cmn/bedrock_models.py), limited to models
# that can call tools while streaming (this page uses converse_stream with
# toolConfig), and to fast and light models: no Opus / Fable, no large-tier models
opt_model_excluded = ("opus", "fable", "nova-pro", "70b", "large")
opt_model_list = [
    model for model in FOUNDATION_MODELS.listed_models()
    if model.is_feature_supported(Feature.STREAMING_TOOL_USE)
    and not any(excluded in model.model_id for excluded in opt_model_excluded)
]

with st.sidebar:
    opt_fm: FoundationModel = st.selectbox(label="Model", options=opt_model_list, index=0, format_func=lambda model: model.display_name, key="model_id")

opt_model_id = opt_fm.model_id

opt_fm_max_tokens = opt_fm.inference_parameters.max_tokens
opt_fm_temperature = opt_fm.inference_parameters.temperature
opt_fm_top_p = opt_fm.inference_parameters.top_p
opt_fm_top_k = opt_fm.inference_parameters.top_k

# Sliders only for the parameters the model accepts; the request builder drops the rest
with st.sidebar:
    opt_temperature = None
    opt_top_p = None
    opt_top_k = None
    if opt_fm_temperature.is_supported:
        opt_temperature = st.slider(label="Temperature", min_value=0.0, max_value=1.0, value=0.1, step=0.1, key="temperature")
    if opt_fm_top_p.is_supported:
        opt_top_p = st.slider(label="Top P", min_value=0.0, max_value=1.0, value=1.0, step=0.1, key="top_p")
    if opt_fm_top_k.is_supported:
        opt_top_k = st.slider(label="Top K", min_value=0, max_value=500, value=250, step=1, key="top_k")
    opt_max_tokens = st.slider(label="Max Tokens", min_value=opt_fm_max_tokens.min_value, max_value=opt_fm_max_tokens.max_value, value=opt_fm_max_tokens.default_value, step=1, key="max_tokens")
    opt_system_msg = st.text_area(label="System Message", value="You are a question and answering chatbot", key="system_msg")

# Docked to the bottom of the page (outside the columns)
prompt = st.chat_input()

with main_col:
    st.markdown("Converse Tool 375")

    if "menu_converse_tool_messages" not in st.session_state:
        st.session_state["menu_converse_tool_messages"] = []

    if "menu_converse_tool_log" not in st.session_state:
        st.session_state["menu_converse_tool_log"] = []  # every tool call: name, input, result


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
                st.markdown(f"{content_text} \n\n:green[Document: {document_name}]" if document_name else content_text)
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

        # Only the parameters the model supports are sent
        inference_settings = InferenceSettings(max_tokens=opt_max_tokens, temperature=opt_temperature, top_p=opt_top_p, top_k=opt_top_k)
        inference_config = build_inference_config(opt_fm, inference_settings)
        additional_model_fields = build_additional_model_fields(opt_fm, inference_settings) or None

        with st.spinner('Processing...'):
            try:
                tool_uses = {}  # contentBlockIndex -> {"toolUseId", "name", "input" (JSON text)}
                call_metrics = None
                llm_calls = []  # invocation metrics of every LLM call in this turn

                response = get_bedrock_runtime(opt_fm.region).converse_stream(
                    modelId=opt_model_id,
                    messages=message_history,
                    system=system_prompts,
                    toolConfig=tool_config,
                    inferenceConfig=inference_config,
                    additionalModelRequestFields=additional_model_fields
                )

                result_text = ""
                # One bubble per turn: the follow-up answer after the tool results streams into it too
                assistant_message = st.chat_message("assistant")
                with assistant_message:
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
                                    print(f"tool_use_id={content_block_tool_use['toolUseId']} tool_use_name={content_block_tool_use['name']}")
                                    tool_uses[content_block_start['contentBlockIndex']] = {
                                        "toolUseId": content_block_tool_use['toolUseId'],
                                        "name": content_block_tool_use['name'],
                                        "input": "",
                                    }

                        if 'contentBlockDelta' in event:
                            content_delta = event['contentBlockDelta']['delta']
                            if 'text' in content_delta:
                                result_text += f"{content_delta['text']}"
                                result_area.write(result_text)
                            if 'toolUse' in content_delta:
                                # Each tool use streams its own input; several can come in one response
                                tool_uses[event['contentBlockDelta']['contentBlockIndex']]["input"] += content_delta['toolUse']['input']

                        if 'messageStop' in event:
                            stop_reason = event['messageStop']['stopReason']
                            if stop_reason == 'end_turn':
                                pass
                            elif "tool_use" == stop_reason:
                                print(f"tool uses: {list(tool_uses.values())}")
                                pass
                            else:
                                stop_reason_display = stop_reason
                                if stop_reason == 'max_tokens':
                                    stop_reason_display = "Insufficient Tokens. Increaes MaxToken Settings."
                                result_text_error = f"{result_text}\n\n:red[Generation Stopped: {stop_reason_display}]"
                                result_area.write(result_text_error)

                        if 'metadata' in event:
                            call_metrics = read_invocation_metrics(event['metadata'])

                        if "internalServerException" in event:
                            exception = event["internalServerException"]
                            result_text += f"\n\n{exception}"
                            result_area.write(result_text)
                        if "modelStreamErrorException" in event:
                            exception = event["modelStreamErrorException"]
                            result_text += f"\n\n{exception}"
                            result_area.write(result_text)
                        if "throttlingException" in event:
                            exception = event["throttlingException"]
                            result_text += f"\n\n{exception}"
                            result_area.write(result_text)
                        if "validationException" in event:
                            exception = event["validationException"]
                            result_text += f"\n\n{exception}"
                            result_area.write(result_text)

                if call_metrics:
                    llm_calls.append(call_metrics)

                if tool_uses:
                    # The model can request several tools in one response; Converse needs a
                    # toolResult for every toolUse, so all of them run (one round per turn)
                    tool_request_content = []
                    tool_result_content = []
                    for tool_use in tool_uses.values():
                        tool_name = tool_use["name"]
                        tool_args_json = json.loads(tool_use["input"] or "{}")
                        tool_request_content.append({
                            "toolUse": {"toolUseId": tool_use["toolUseId"], "name": tool_name, "input": tool_args_json}
                        })

                        matched_tool = next((tool for tool in tools if tool.matches(tool_name)), None)
                        try:
                            if matched_tool is None:
                                expr_result = {"error": f"Unknown tool: {tool_name}"}
                            else:
                                expr_result = matched_tool.invoke(tool_args_json.get('expression'), tool_args=tool_args_json)
                        except Exception as err:
                            logger.exception("Tool %s failed", tool_name)
                            expr_result = {"error": f"{type(err).__name__}: {err}"}

                        tool_result_content.append({
                            "toolResult": {
                                "toolUseId": tool_use["toolUseId"],
                                "content": [{"json": {"expr_result": expr_result}}]
                            }
                        })

                        # Tool request / result go to the Info Panel's Tool Use Log, not the chat
                        st.session_state.menu_converse_tool_log.append({
                            "tool_name": tool_name,
                            "tool_use_id": tool_use["toolUseId"],
                            "input": tool_args_json,
                            "result": expr_result,
                        })

                    tool_request_message = {"role": "assistant", "content": tool_request_content}
                    tool_result_message = {"role": "user", "content": tool_result_content}

                    # Full history (ends with message_user_latest) so the follow-up call keeps earlier turns
                    messages = message_history + [tool_request_message, tool_result_message]

                    follow_up_text, follow_up_metrics, follow_up_tool_uses = invoke_llm(messages, system_prompts, tool_config, inference_config, additional_model_fields, assistant_message)
                    if follow_up_metrics:
                        llm_calls.append(follow_up_metrics)
                    # One tool round per turn: further tool requests are logged, not run
                    for tool_use in follow_up_tool_uses:
                        st.session_state.menu_converse_tool_log.append({
                            "tool_name": tool_use["name"],
                            "tool_use_id": tool_use["toolUseId"],
                            "input": tool_use["input"],
                            "result": "Not run: this prototype runs one round of tool calls per turn (3_7_5_converse_tool_v3 loops)",
                        })
                    # Save the answer that came after the tool result, not just the text before the tool call
                    result_text = "\n\n".join(text for text in (result_text, follow_up_text) if text.strip())

                st.session_state.menu_converse_tool_last_metrics = {"model": opt_fm.display_name, "calls": llm_calls}

                message_assistant_latest = {"role": "assistant", "content": [{"text": result_text or "(no response)"}]}

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

# Right side Info Panel, pinned while the conversation scrolls (same CSS as
# 3_5_13_converse_demo: the COLUMN containing the keyed container is made sticky)
st.html(
    """
<style>
[data-testid="stColumn"]:has(.st-key-info_panel_sticky) {
    position: sticky;
    top: 3.75rem; /* clear the Streamlit header */
    align-self: flex-start;
    height: fit-content;
    max-height: calc(100vh - 3.75rem - 10rem);
    overflow-y: auto;
}
</style>
"""
)

with right_sidebar_col, st.container(key="info_panel_sticky"):
    st.markdown("### Info Panel")

    with st.container(border=True):
        st.markdown("**Statistics**")
        st.metric("Messages", len(st.session_state.menu_converse_tool_messages))
        st.metric("Limit", MAX_MESSAGES)

    # Invocation metrics of the last turn: one LLM call, or two when a tool was used
    last_metrics = st.session_state.get("menu_converse_tool_last_metrics")
    if last_metrics and last_metrics["calls"]:
        calls = last_metrics["calls"]
        with st.container(border=True):
            st.markdown("**Invocation Metrics**")
            st.caption(f"{last_metrics['model']} | {len(calls)} LLM call(s)")
            with st.container(horizontal=True):
                st.metric("Tokens In", f"{sum(c['input_tokens'] for c in calls):,}")
                st.metric("Tokens Out", f"{sum(c['output_tokens'] for c in calls):,}")
            with st.container(horizontal=True):
                st.metric("Total", f"{sum(c['total_tokens'] for c in calls):,}")
                st.metric("Latency", f"{sum(c['latency_ms'] for c in calls):,} ms")
            if len(calls) > 1:
                for number, call in enumerate(calls, 1):
                    st.caption(f"Call {number}: in={call['input_tokens']:,} out={call['output_tokens']:,} latency={call['latency_ms']:,} ms")

    with st.container(border=True):
        st.markdown("**Tool Use Log**")
        tool_log = st.session_state.menu_converse_tool_log
        if not tool_log:
            st.caption("No tool calls yet")
        # Newest first; only the latest entry starts expanded
        for number in range(len(tool_log), 0, -1):
            entry = tool_log[number - 1]
            with st.expander(f"{number}. {entry['tool_name']}", expanded=number == len(tool_log)):
                st.caption("Input")
                st.code(json.dumps(entry["input"], indent=2, default=str), language="json")
                st.caption("Result")
                st.code(json.dumps(entry["result"], indent=2, default=str), language="json")

    with st.container(border=True):
        st.markdown("**Available Tools**")
        for tool in tools:
            st.caption(tool.definition['toolSpec']['name'])
