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
- Capped tool loop: runs every tool the model requests (several per response), then calls the
  model again, up to MAX_TOOL_ROUNDS rounds per turn
- Inline stream parsing: AWS Bedrock event stream parsing logic embedded directly in UI code
- Manual tool matching: Iterates through tool list to find matching tool by name
- Simple metrics: Displays token counts per LLM call without accumulation across turns
- Static system prompt: Tool descriptions must be manually maintained
- No rich rendering: Tool results displayed as JSON dumps only
- Tightly coupled: Business logic mixed with Streamlit UI code, cannot be tested independently

LIMITATIONS
-----------
The tool loop stops after MAX_TOOL_ROUNDS rounds; tool requests beyond that are logged in the
Info Panel as not run. Stream parsing, the loop and the UI still live inline in this page, and
tool results have no rich rendering. Version 3.7.5 moves the loop into ConversationManager.

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
MAX_TOOL_ROUNDS = 5  # rounds of tool calls per turn before the tool loop stops
# Text block added to a user message (once the user has toggled tools) recording which tools
# were available for it; kept in history for the model, hidden in the chat
TOOL_AVAILABILITY_PREFIX = "[Tools available for this message:"

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
# The tools sent to the model (tool_config) are the ones ticked in the sidebar

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

def parse_tool_input(tool_input):
    """A tool use's streamed JSON input as a dict (Converse needs a dict even if the JSON is bad)."""
    try:
        return json.loads(tool_input or "{}")
    except json.JSONDecodeError:
        logger.warning("Tool input is not valid JSON: %s", tool_input)
        return {"invalid_json": tool_input}

def invoke_llm(message_history, system_prompts, tool_config, inference_config, additional_model_fields, assistant_message):
    """
    One converse_stream call, streamed into the turn's assistant bubble. Returns a dict:
        text              - response text
        metrics           - invocation metrics, or None
        tool_uses         - requested tools: {"toolUseId", "name", "input" (JSON text)}
        assistant_content - the response as Converse content blocks (reasoning, text, toolUse),
                            sent back as the assistant message together with the tool results
    """
    call_metrics = None
    result_text = ""
    content_blocks = {}  # contentBlockIndex -> block being streamed (reasoningContent / text / toolUse)
    try:
        request = dict(
            modelId=opt_model_id,
            messages=message_history,
            system=system_prompts,
            inferenceConfig=inference_config,
            additionalModelRequestFields=additional_model_fields
        )
        if tool_config:  # None when every tool is unticked
            request["toolConfig"] = tool_config
        response = get_bedrock_runtime(opt_fm.region).converse_stream(**request)

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
                            content_blocks[content_block_start['contentBlockIndex']] = {"toolUse": {
                                "toolUseId": content_block_tool_use['toolUseId'],
                                "name": content_block_tool_use['name'],
                                "input": "",
                            }}

                if 'contentBlockDelta' in event:
                    block_index = event['contentBlockDelta']['contentBlockIndex']
                    content_delta = event['contentBlockDelta']['delta']
                    print(f"content_delta {content_delta}")
                    if 'text' in content_delta:
                        result_text += f"{content_delta['text']}"
                        result_area.write(result_text)
                        content_blocks.setdefault(block_index, {"text": ""})["text"] += content_delta['text']
                    if 'toolUse' in content_delta:
                        # Each tool use streams its own input; several can come in one response
                        content_blocks[block_index]["toolUse"]["input"] += content_delta['toolUse']['input']
                    if 'reasoningContent' in content_delta:
                        # Thinking must go back unchanged (with its signature) alongside the tool results
                        reasoning_delta = content_delta['reasoningContent']
                        if 'redactedContent' in reasoning_delta:
                            content_blocks[block_index] = {"reasoningContent": {"redactedContent": reasoning_delta['redactedContent']}}
                        else:
                            reasoning_text = content_blocks.setdefault(
                                block_index, {"reasoningContent": {"reasoningText": {"text": ""}}}
                            )["reasoningContent"]["reasoningText"]
                            reasoning_text["text"] += reasoning_delta.get('text', '')
                            if 'signature' in reasoning_delta:
                                reasoning_text["signature"] = reasoning_delta['signature']

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

    tool_uses = []
    assistant_content = []
    for block_index in sorted(content_blocks):
        block = content_blocks[block_index]
        if "toolUse" in block:
            tool_use = block["toolUse"]
            tool_uses.append(tool_use)
            assistant_content.append({"toolUse": {
                "toolUseId": tool_use["toolUseId"],
                "name": tool_use["name"],
                "input": parse_tool_input(tool_use["input"]),
            }})
        elif "text" in block:
            if block["text"].strip():  # Converse rejects blank text blocks
                assistant_content.append(block)
        else:
            assistant_content.append(block)

    return {"text": result_text, "metrics": call_metrics, "tool_uses": tool_uses, "assistant_content": assistant_content}

def without_tool_blocks(history):
    """
    History for calls that send no toolConfig (every tool unticked): Converse rejects toolUse /
    toolResult blocks then. Each one becomes a short text note instead of being dropped, so the
    model still sees which tools ran and what they returned, and roles still alternate.
    """
    plain_history = []
    for msg in history:
        content = []
        for block in msg["content"]:
            if "toolUse" in block:
                tool_use = block["toolUse"]
                content.append({"text": f"[Called tool {tool_use['name']} with {json.dumps(tool_use['input'], default=str)}]"})
            elif "toolResult" in block:
                result = json.dumps(block["toolResult"]["content"], default=str)
                content.append({"text": f"[Tool result: {result[:2000]}]"})
            else:
                content.append(block)
        plain_history.append({"role": msg["role"], "content": content})
    return plain_history

def tool_availability_note(enabled_tools):
    """
    System prompt note once the user has unticked any tool in this conversation, so the model
    doesn't take earlier turns (made with a different set of tools) for its own mistakes.
    None while every tool has stayed on.
    """
    if len(enabled_tools) < len(tools):
        st.session_state.menu_converse_tool_availability_changed = True
    if not st.session_state.get("menu_converse_tool_availability_changed"):
        return None
    tool_name = lambda tool: tool.definition['toolSpec']['name']
    enabled = ", ".join(tool_name(tool) for tool in enabled_tools) or "none"
    disabled = ", ".join(tool_name(tool) for tool in tools if tool not in enabled_tools) or "none"
    return (
        "The user chooses which tools are available, and this can change between turns. "
        f"Available now: {enabled}. Disabled now: {disabled}. User messages end with a "
        f"'{TOOL_AVAILABILITY_PREFIX} ...]' note recording the tools available when they were sent "
        "(messages without the note had every tool available). Earlier turns reflect the tools "
        "available at that time: tool calls and results in the history were real, and a tool "
        "described as disabled earlier was disabled then. Do not treat either as a mistake."
    )

def build_display_messages(history):
    """
    The messages the chat shows: [role, text, document name or None] per bubble.
    History holds whole turns (tool requests / results too); only text is shown,
    one assistant bubble per turn like the live view. Also used for the message
    count and the history limit, so both match what the user sees.
    """
    display_messages = []
    for msg in history:
        texts = [block["text"] for block in msg["content"]
                 if "text" in block and not block["text"].startswith(TOOL_AVAILABILITY_PREFIX)]
        if not texts:
            continue  # toolUse / toolResult only
        text = "\n\n".join(texts)
        if msg["role"] == "assistant" and display_messages and display_messages[-1][0] == "assistant":
            display_messages[-1][1] += "\n\n" + text
            continue
        document_name = next((block["document"]["name"] for block in msg["content"] if "document" in block), None)
        display_messages.append([msg["role"], text, document_name])
    return display_messages


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

    # Tool selection lives in the left sidebar; the right panel is per-turn output.
    # Only ticked tools are offered to the model; changes apply from the next message.
    with st.container(border=True):
        st.markdown("**Available Tools**")
        enabled_tools = []
        for tool in tools:
            tool_spec = tool.definition['toolSpec']
            if st.checkbox(tool_spec['name'], value=True, key=f"tool_enabled_{tool_spec['name']}", help=tool_spec['description']):
                enabled_tools.append(tool)

tool_config = {"tools": [tool.definition for tool in enabled_tools]} if enabled_tools else None

# Docked to the bottom of the page (outside the columns), with the message count just above it.
# The count is filled in at the end of the script so it includes the turn that just ran.
with st.bottom:
    message_count_area = st.empty()
    prompt = st.chat_input()

with main_col:
    st.markdown("Converse Tool 375")

    if "menu_converse_tool_messages" not in st.session_state:
        st.session_state["menu_converse_tool_messages"] = []

    if "menu_converse_tool_log" not in st.session_state:
        st.session_state["menu_converse_tool_log"] = []  # every tool call: name, input, result


    for role, content_text, document_name in build_display_messages(st.session_state.menu_converse_tool_messages):
        with st.chat_message(role):
            st.markdown(f"{content_text} \n\n:green[Document: {document_name}]" if document_name else content_text)

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
        availability_note = tool_availability_note(enabled_tools)
        if availability_note:
            system_prompts.append({"text": availability_note})
            # Per-turn record in history (message_user_latest is the last message of message_history)
            enabled_names = ", ".join(tool.definition['toolSpec']['name'] for tool in enabled_tools) or "none"
            message_user_latest["content"].append({"text": f"{TOOL_AVAILABILITY_PREFIX} {enabled_names}]"})

        # Only the parameters the model supports are sent
        inference_settings = InferenceSettings(max_tokens=opt_max_tokens, temperature=opt_temperature, top_p=opt_top_p, top_k=opt_top_k)
        inference_config = build_inference_config(opt_fm, inference_settings)
        additional_model_fields = build_additional_model_fields(opt_fm, inference_settings) or None

        with st.spinner('Processing...'):
            try:
                llm_calls = []      # invocation metrics of every LLM call in this turn
                # With every tool unticked no toolConfig is sent, so earlier tool blocks must go
                messages = message_history if tool_config else without_tool_blocks(message_history)
                # One bubble per turn: every call in the tool loop streams its text into it
                assistant_message = st.chat_message("assistant")

                # Tool loop: run the requested tools and call the model again until it answers
                # without tools, at most MAX_TOOL_ROUNDS rounds of tool calls per turn
                for tool_round in range(MAX_TOOL_ROUNDS + 1):
                    call = invoke_llm(messages, system_prompts, tool_config, inference_config, additional_model_fields, assistant_message)
                    if call["metrics"]:
                        llm_calls.append(call["metrics"])
                    if not call["tool_uses"]:
                        break

                    if tool_round == MAX_TOOL_ROUNDS:
                        # Safety stop: log what the model still asked for
                        for tool_use in call["tool_uses"]:
                            st.session_state.menu_converse_tool_log.append({
                                "tool_name": tool_use["name"],
                                "tool_use_id": tool_use["toolUseId"],
                                "input": tool_use["input"],
                                "result": f"Not run: reached the limit of {MAX_TOOL_ROUNDS} tool rounds per turn",
                            })
                        break

                    # Run every requested tool; Converse needs a toolResult for each toolUse
                    tool_result_content = []
                    for tool_use in call["tool_uses"]:
                        tool_name = tool_use["name"]
                        tool_args_json = parse_tool_input(tool_use["input"])

                        matched_tool = next((tool for tool in enabled_tools if tool.matches(tool_name)), None)
                        try:
                            if matched_tool is None:
                                is_known_tool = any(tool.matches(tool_name) for tool in tools)
                                expr_result = {"error": f"Tool {tool_name} is disabled" if is_known_tool else f"Unknown tool: {tool_name}"}
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

                    # Full history (ends with message_user_latest) plus this turn's tool rounds
                    messages = messages + [
                        {"role": "assistant", "content": call["assistant_content"]},
                        {"role": "user", "content": tool_result_content},
                    ]

                st.session_state.menu_converse_tool_last_metrics = {"model": opt_fm.display_name, "calls": llm_calls}

                # Save the whole turn, tool requests / results included, so later turns can see
                # which tools already ran (text-only history made the model redo earlier tool calls).
                # Thinking is only needed inside an active tool loop, so it is not kept.
                turn_messages = messages[len(message_history) - 1:]  # message_user_latest + tool rounds
                # Final call: text only (if the loop hit its limit, its toolUse blocks have no results)
                final_content = [block for block in call["assistant_content"] if "text" in block]
                turn_messages.append({"role": "assistant", "content": final_content or [{"text": "(no response)"}]})
                for turn_message in turn_messages:
                    st.session_state.menu_converse_tool_messages.append({
                        "role": turn_message["role"],
                        "content": [block for block in turn_message["content"] if "reasoningContent" not in block],
                    })

                # Trim from the front; history must start with a user text message, never mid tool exchange
                menu_converse_tool_messages = st.session_state.menu_converse_tool_messages
                while menu_converse_tool_messages and (
                    len(build_display_messages(menu_converse_tool_messages)) > MAX_MESSAGES
                    or menu_converse_tool_messages[0]["role"] != "user"
                    or not any("text" in block for block in menu_converse_tool_messages[0]["content"])
                ):
                    del menu_converse_tool_messages[0]

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

def clear_conversation():
    """Clear the conversation and everything the Info Panel shows for it"""
    st.session_state.menu_converse_tool_messages = []
    st.session_state.menu_converse_tool_log = []
    st.session_state.pop("menu_converse_tool_last_metrics", None)
    st.session_state.pop("menu_converse_tool_availability_changed", None)

# Message count and clear button above the chat input (same style as 3_5_13_converse_demo);
# filled last so the count includes the turn that just ran. Counts the messages the chat
# shows (tool requests / results are in history but not counted).
with message_count_area.container(horizontal=True, horizontal_alignment="right", vertical_alignment="center", gap="xxsmall"):
    st.markdown(
        f":violet[**{len(build_display_messages(st.session_state.menu_converse_tool_messages))}/{MAX_MESSAGES}**]",
        help="Messages in conversation history / limit",
    )
    st.button(":material/delete_history:", type="tertiary", help="Clear Conversation", key="clear_conversation", on_click=clear_conversation)
