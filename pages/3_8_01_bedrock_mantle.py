import streamlit as st
import json
import logging
import cmn_settings
from PIL import Image
import io
import base64
import pandas as pd
from typing import Optional, Callable, Any
from dataclasses import dataclass, field

from anthropic import AnthropicBedrockMantle
from anthropic.types import Message, ContentBlock, TextBlock, ToolUseBlock

from cmn.view.mime_constants import mime_mapping_image, mime_mapping_document
from cmn.view import CONVERSE_TOOL_GUIDE
from cmn.view.processor.file_uploader_chat import render_file_uploader

from cmn.tools.tool import (
    AwsDocsBedrockConverseTool,
    WikipediaBedrockConverseTool,
    UrlContentBedrockConverseTool,
    AcronymBedrockConverseTool,
    SalesBedrockConverseTool,
    SalesKpiBedrockConverseTool,
    SalesForecastBedrockConverseTool,
    SalesAnomalyBedrockConverseTool,
    DateTimeBedrockConverseTool,
    CalculatorBedrockConverseTool,
    ChartBedrockConverseTool,
    ProductBedrockConverseTool,
    PdfBedrockConverseTool,
    PptxBedrockConverseTool,
)

from cmn.tools.renderer import (
    RendererRegistry,
    ChartToolRenderer,
    ProductToolRenderer,
    SalesKpiToolRenderer,
    SalesAnomalyToolRenderer,
    SalesForecastToolRenderer,
    PptxToolRenderer,
    PdfToolRenderer,
)
from cmn.tools.tool import ToolRegistry

AWS_REGION = cmn_settings.AWS_REGION
MAX_MESSAGES = 100 * 2

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

################################################################################
# SECTION: InvocationStat Dataclass
################################################################################

@dataclass
class InvocationStat:
    input_tokens:  int = 0
    output_tokens: int = 0
    total_tokens:  int = 0
    latency_ms:    int = 0
    llm_calls:     int = 0
    tools_called:  list = field(default_factory=list)

    def accumulate(self, usage: dict, latency_ms: int = 0) -> None:
        """Accumulate metrics from Anthropic message usage."""
        self.input_tokens  += usage.get('input_tokens', 0)
        self.output_tokens += usage.get('output_tokens', 0)
        self.total_tokens  += (usage.get('input_tokens', 0) + 
                               usage.get('output_tokens', 0))
        self.latency_ms    += latency_ms
        self.llm_calls     += 1

    def record_tool(self, tool_name: str) -> None:
        self.tools_called.append(tool_name)

    def as_markdown(self) -> str:
        lines = [
            f"🔢 in={self.input_tokens} out={self.output_tokens} "
            f"total={self.total_tokens} latency={self.latency_ms}ms "
            f"calls={self.llm_calls}"
        ]
        if self.tools_called:
            lines.append(f"🔧 tools: {', '.join(f'`{n}`' for n in self.tools_called)}")
        return "  \n".join(lines)


################################################################################
# SECTION: Mantle Conversation Manager
################################################################################

class MantleConversationManager:
    """
    Orchestrates multi-turn conversations using Anthropic Bedrock Mantle.
    Handles tool-use loops with streaming support.
    """

    def __init__(
        self,
        mantle_client: AnthropicBedrockMantle,
        tool_registry: Optional[ToolRegistry],
        model_id: str,
        max_tokens: int,
        temperature: float,
        system_prompt: str,
    ):
        self.client = mantle_client
        self.registry = tool_registry
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.system_prompt = system_prompt

    def run(
        self,
        message_history: list,
        on_text_delta: Optional[Callable[[str], None]] = None,
        on_message_complete: Optional[Callable[[dict, int], None]] = None,
        on_tool_invoked: Optional[Callable[[str, dict, Any], None]] = None,
    ) -> tuple[str, dict]:
        """
        Run conversation with tool loop support.
        Returns: (final_text, usage_dict)
        """
        messages = self._convert_messages(message_history)
        accumulated_text = ""
        total_usage = {'input_tokens': 0, 'output_tokens': 0}

        while True:
            import time
            start_time = time.time()

            # Build API call parameters
            api_params = {
                "model": self.model_id,
                "max_tokens": self.max_tokens,
                #"temperature": self.temperature,
                "system": self.system_prompt,
                "messages": messages,
                "stream": True,
            }

            # Only add tools if registry exists and has get_anthropic_tools method
            if self.registry and hasattr(self.registry, 'get_anthropic_tools'):
                tools = self.registry.get_anthropic_tools()
                if tools:
                    api_params["tools"] = tools

            # Call Anthropic API with streaming
            response = self.client.messages.create(**api_params)

            # Process stream
            current_text = ""
            tool_uses = []

            for event in response:
                if event.type == "content_block_start":
                    if hasattr(event, 'content_block'):
                        block = event.content_block
                        if block.type == "tool_use":
                            tool_uses.append({
                                'id': block.id,
                                'name': block.name,
                                'input': {}
                            })

                elif event.type == "content_block_delta":
                    delta = event.delta
                    if delta.type == "text_delta":
                        chunk = delta.text
                        current_text += chunk
                        accumulated_text += chunk
                        if on_text_delta:
                            on_text_delta(chunk)

                    elif delta.type == "input_json_delta":
                        # Accumulate tool input
                        if tool_uses:
                            partial_json = delta.partial_json
                            # Merge into the last tool use
                            tool_uses[-1]['input_partial'] = tool_uses[-1].get('input_partial', '') + partial_json

                elif event.type == "message_delta":
                    # Update usage stats
                    if hasattr(event, 'usage'):
                        total_usage['output_tokens'] += event.usage.output_tokens

                elif event.type == "message_start":
                    if hasattr(event, 'message') and hasattr(event.message, 'usage'):
                        total_usage['input_tokens'] += event.message.usage.input_tokens

            latency_ms = int((time.time() - start_time) * 1000)

            # Finalize tool inputs
            for tool_use in tool_uses:
                if 'input_partial' in tool_use:
                    try:
                        tool_use['input'] = json.loads(tool_use['input_partial'])
                    except json.JSONDecodeError:
                        tool_use['input'] = {}
                    del tool_use['input_partial']

            # Notify callback
            if on_message_complete:
                on_message_complete(total_usage, latency_ms)

            # Check if we need to process tools
            if not tool_uses or not self.registry:
                return accumulated_text, total_usage

            # Add assistant message with tool uses
            messages.append({
                "role": "assistant",
                "content": self._build_assistant_content(current_text, tool_uses)
            })

            # Execute tools and build tool results
            tool_results = []
            for tool_use in tool_uses:
                tool_name = tool_use['name']
                tool_input = tool_use['input']

                try:
                    result = self.registry.invoke(tool_name, tool_input)

                    if on_tool_invoked:
                        on_tool_invoked(tool_name, tool_input, result)

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tool_use['id'],
                        "content": str(result)
                    })
                except Exception as e:
                    logger.error(f"Tool {tool_name} failed: {e}")
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tool_use['id'],
                        "content": f"Error: {str(e)}",
                        "is_error": True
                    })

            # Add user message with tool results
            messages.append({
                "role": "user",
                "content": tool_results
            })

    def _convert_messages(self, message_history: list) -> list:
        """Convert internal message format to Anthropic format."""
        converted = []
        for msg in message_history:
            role = msg["role"]
            content = msg["content"]

            # Simple text message
            if len(content) == 1 and "text" in content[0]:
                converted.append({
                    "role": role,
                    "content": content[0]["text"]
                })
            else:
                # Complex content with images/documents
                anthropic_content = []
                for item in content:
                    if "text" in item:
                        anthropic_content.append({
                            "type": "text",
                            "text": item["text"]
                        })
                    elif "image" in item:
                        img_data = item["image"]
                        anthropic_content.append({
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": f"image/{img_data['format']}",
                                "data": base64.b64encode(img_data['source']['bytes']).decode('utf-8')
                            }
                        })
                    elif "document" in item:
                        doc_data = item["document"]
                        # Anthropic doesn't support documents directly, convert to text
                        anthropic_content.append({
                            "type": "text",
                            "text": f"[Document: {doc_data.get('name', 'unknown')}]"
                        })

                converted.append({
                    "role": role,
                    "content": anthropic_content
                })

        return converted

    def _build_assistant_content(self, text: str, tool_uses: list) -> list:
        """Build assistant content with text and tool uses."""
        content = []
        if text:
            content.append({"type": "text", "text": text})
        for tool_use in tool_uses:
            content.append({
                "type": "tool_use",
                "id": tool_use['id'],
                "name": tool_use['name'],
                "input": tool_use['input']
            })
        return content


################################################################################
# SECTION: File / Media Utilities
################################################################################

def build_user_message(
    prompt: str,
    uploaded_file=None,
    uploaded_file_bytes=None,
    uploaded_file_type: str = None,
    uploaded_file_key: str = None,
) -> dict:
    """
    Construct the user content list, optionally attaching an image or document.
    """
    content = [{"text": prompt}]

    if uploaded_file and uploaded_file_bytes and uploaded_file_type:
        if uploaded_file_type in mime_mapping_image:
            content.append({
                "image": {
                    "format": mime_mapping_image[uploaded_file_type],
                    "source": {"bytes": uploaded_file_bytes},
                }
            })
        elif uploaded_file_type in mime_mapping_document:
            content.append({
                "document": {
                    "format": mime_mapping_document[uploaded_file_type],
                    "name":   uploaded_file_key or "uploaded_document",
                    "source": {"bytes": uploaded_file_bytes},
                }
            })

    return {"role": "user", "content": content}


################################################################################
# SECTION: Shared Streamlit Resources
################################################################################

@st.cache_resource
def get_mantle_client():
    return AnthropicBedrockMantle(aws_region=AWS_REGION)


@st.cache_resource
def get_renderer_registry():
    return RendererRegistry([
        ChartToolRenderer(),
        ProductToolRenderer(),
        SalesKpiToolRenderer(),
        SalesAnomalyToolRenderer(),
        SalesForecastToolRenderer(),
        PptxToolRenderer(),
        PdfToolRenderer(),
    ])


@st.cache_resource
def get_tool_registry():
    # Return None for now - we'll add tools later
    return None
    # Uncomment when ready to add tools:
    # return ToolRegistry([
    #     CalculatorBedrockConverseTool(),
    #     AcronymBedrockConverseTool(),
    #     UrlContentBedrockConverseTool(),
    #     WikipediaBedrockConverseTool(),
    #     AwsDocsBedrockConverseTool(),
    #     DateTimeBedrockConverseTool(),
    #     SalesBedrockConverseTool(),
    #     ProductBedrockConverseTool(),
    #     ChartBedrockConverseTool(),
    #     SalesKpiBedrockConverseTool(),
    #     SalesForecastBedrockConverseTool(),
    #     SalesAnomalyBedrockConverseTool(),
    #     PptxBedrockConverseTool(),
    #     PdfBedrockConverseTool(),
    # ])


mantle_client = get_mantle_client()
tool_registry = get_tool_registry()
renderer_registry = get_renderer_registry()


################################################################################
# SECTION: System Prompt
################################################################################

def build_default_system_prompt(registry: Optional[ToolRegistry]) -> str:
    base_prompt = "You are a helpful AI assistant."

    if registry and hasattr(registry, 'build_tool_summary'):
        tool_summary = registry.build_tool_summary()
        return "\n\n".join(filter(None, [
            "You are a BI analyst assistant.",
            tool_summary,
            "Call tools ONE AT A TIME. Wait for each result before calling the next.",
        ]))

    return base_prompt


OPT_SYSTEM_MSG_DEFAULT = build_default_system_prompt(tool_registry)


################################################################################
# SECTION: Streamlit Sidebar / Options
################################################################################

opt_model_id_list = [
    #"global.anthropic.claude-opus-4-7",
    "anthropic.claude-opus-4-7",
    #"global.anthropic.claude-sonnet-4-20250514-v1:0",
    #"anthropic.claude-sonnet-4-20250514-v1:0",
    #"anthropic.claude-opus-4-6",
    #"anthropic.claude-sonnet-4-6",
    #"anthropic.claude-sonnet-4-5",
    #"anthropic.claude-mythos-preview",
]

with st.sidebar:
    opt_model_id = st.selectbox(
        "Model ID", 
        opt_model_id_list, 
        index=0, 
        key="bedrock_mantle_model_id"
    )
    # opt_temperature = st.slider(
    #     "Temperature", 
    #     0.0, 1.0, 0.1, 0.1, 
    #     key="bedrock_mantle_temperature"
    # )
    opt_max_tokens = st.slider(
        "Max Tokens", 
        0, 32000, 4096, 1, 
        key="bedrock_mantle_max_tokens"
    )
    opt_system_msg = st.text_area(
        "System Message", 
        OPT_SYSTEM_MSG_DEFAULT, 
        key="bedrock_mantle_system_msg"
    )
    opt_show_metrics = st.checkbox(
        "Show Invocation Metrics", 
        value=False, 
        key="bedrock_mantle_show_metrics"
    )

    if tool_registry:
        with st.expander("Tools"):
            st.markdown(f"Tools: {', '.join(tool_registry.tool_names)}")
    else:
        st.info("Tools disabled - basic chat mode")


################################################################################
# SECTION: Streamlit Page Setup + Session State
################################################################################

with st.container(horizontal=True, vertical_alignment="center"):
    st.markdown("💬 Bedrock Mantle Converse Tool")
    show_examples = st.toggle("Examples", value=False, key="bedrock_mantle_show_examples")

if show_examples:
    st.info(CONVERSE_TOOL_GUIDE)

if "bedrock_mantle_messages" not in st.session_state:
    st.session_state.bedrock_mantle_messages = []

if "bedrock_mantle_invocation_stats" not in st.session_state:
    st.session_state.bedrock_mantle_invocation_stats = []

if len(st.session_state.bedrock_mantle_invocation_stats) != len(st.session_state.bedrock_mantle_messages):
    st.session_state.bedrock_mantle_invocation_stats = [None] * len(st.session_state.bedrock_mantle_messages)

if "bedrock_mantle_uploader_key" not in st.session_state:
    st.session_state.bedrock_mantle_uploader_key = 0

st.markdown(f"{len(st.session_state.bedrock_mantle_messages)}/{MAX_MESSAGES}")


################################################################################
# SECTION: Render Message History
################################################################################

for idx, msg in enumerate(st.session_state.bedrock_mantle_messages):
    with st.chat_message(msg["role"]):
        contents = msg["content"]
        text = contents[0].get("text", "")

        if msg["role"] == "user" and len(contents) > 1:
            extra = contents[1]
            if "document" in extra:
                doc_name = extra["document"].get("name", "")
                st.markdown(f"{text}\n\n:green[Document: {doc_name}]")
            elif "image" in extra:
                st.markdown(f"{text}\n\n:green[Image attached]")
            else:
                st.markdown(text)
        else:
            st.markdown(text)

        if opt_show_metrics and msg["role"] == "assistant":
            stat = st.session_state.bedrock_mantle_invocation_stats[idx]
            if stat is not None:
                st.caption(stat.as_markdown())


################################################################################
# SECTION: File Uploader Widget
################################################################################

uploaded_file, file_bytes, file_key, file_type, file_preview = render_file_uploader(
    st.session_state.bedrock_mantle_uploader_key
)


################################################################################
# SECTION: Chat Input + Conversation Execution
################################################################################

prompt = st.chat_input(key="bedrock_mantle_chat_input")

if prompt:
    # Build user message
    user_message = build_user_message(
        prompt=prompt,
        uploaded_file=uploaded_file,
        uploaded_file_bytes=file_bytes,
        uploaded_file_type=file_type,
        uploaded_file_key=file_key,
    )

    message_history = st.session_state.bedrock_mantle_messages.copy()
    message_history.append(user_message)

    st.chat_message("user").write(prompt)

    # Per-turn accumulators
    accumulated = {"text": ""}
    turn_stat = InvocationStat()

    with st.chat_message("assistant"):
        result_area = st.empty()
        result_container = st.container(border=False)

        # Callbacks
        def on_text_delta(chunk: str):
            """Fired for every streamed text chunk."""
            accumulated["text"] += chunk
            result_area.markdown(accumulated["text"])

        def on_message_complete(usage: dict, latency_ms: int):
            """Fired after each complete LLM response."""
            turn_stat.accumulate(usage, latency_ms)

        def on_tool_invoked(tool_name: str, tool_args: dict, tool_result: Any):
            """Fired after each tool is executed."""
            turn_stat.record_tool(tool_name)
            accumulated["text"] += (
                f"\n\n:blue[🔧 **Tool:** `{tool_name}`]\n"
                f"```json\n{json.dumps(tool_args, indent=2)}\n```\n"
                f"**Result:** `{tool_result}`\n\n"
            )
            result_area.markdown(accumulated["text"])
            if renderer_registry:
                renderer_registry.render(tool_name, tool_args, tool_result, result_container)

        manager = MantleConversationManager(
            mantle_client=mantle_client,
            tool_registry=tool_registry,
            model_id=opt_model_id,
            max_tokens=opt_max_tokens,
            temperature=0.0,
            system_prompt=opt_system_msg,
        )

        with st.spinner("Processing...", show_time=True, width="content"):
            try:
                final_text, final_usage = manager.run(
                    message_history,
                    on_text_delta=on_text_delta,
                    on_message_complete=on_message_complete,
                    on_tool_invoked=on_tool_invoked,
                )
            except Exception as e:
                st.error(f"Error: {str(e)}")
                logger.exception("Error in conversation manager")
                st.stop()

        # Show live metrics
        if opt_show_metrics:
            st.caption(turn_stat.as_markdown())

    # Persist to session state
    assistant_message = {
        "role": "assistant",
        "content": [{"text": accumulated["text"]}],
    }
    st.session_state.bedrock_mantle_messages.append(user_message)
    st.session_state.bedrock_mantle_invocation_stats.append(None)
    st.session_state.bedrock_mantle_messages.append(assistant_message)
    st.session_state.bedrock_mantle_invocation_stats.append(turn_stat)

    # Trim history to MAX_MESSAGES
    msgs = st.session_state.bedrock_mantle_messages
    stats = st.session_state.bedrock_mantle_invocation_stats
    if len(msgs) > MAX_MESSAGES:
        excess = len(msgs) - MAX_MESSAGES
        del msgs[0:excess]
        del stats[0:excess]

    # Reset uploader after successful submission
    if uploaded_file:
        st.session_state.bedrock_mantle_uploader_key += 1
        st.rerun()