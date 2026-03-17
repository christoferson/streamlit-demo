import streamlit as st
import boto3
import json
import logging
import cmn_settings
import cmn_auth
import os
from PIL import Image
import io
import base64
import pandas as pd
from botocore.exceptions import BotoCoreError, ClientError

from cmn.bedrock_converse_tools import CalculatorBedrockConverseTool
from cmn.bedrock_converse_tools_acronym import AcronymBedrockConverseTool
from cmn.bedrock_converse_tools_url import UrlContentBedrockConverseTool
from cmn.bedrock_converse_tools_wikipedia import WikipediaBedrockConverseTool
from cmn.bedrock_converse_tools_datetime import DateTimeBedrockConverseTool
from cmn.bedrock_converse_tools_sales import SalesBedrockConverseTool
from cmn.bedrock_converse_tools_product import ProductBedrockConverseTool
from cmn.bedrock_converse_tools_chart import ChartBedrockConverseTool

AWS_REGION = cmn_settings.AWS_REGION
MAX_MESSAGES = 100 * 2

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

################################################################################
# SECTION: ToolInvocation Dataclass
################################################################################

from dataclasses import dataclass, field
from typing import Any, Optional, Callable

STOP_REASON_MESSAGES = {
    "max_tokens":           "Insufficient Tokens. Increase MaxToken Settings.",
    "guardrail_intervened": "Response blocked by guardrail.",
    "content_filtered":     "Content was filtered.",
}

_EXCEPTION_EVENTS = [
    "internalServerException",
    "modelStreamErrorException",
    "throttlingException",
    "validationException",
    "serviceUnavailableException",
]

@dataclass
class ToolInvocation:
    """Captures a single tool call requested by the model."""
    tool_use_id:    str = None
    tool_name:      str = None
    tool_input_raw: str = ""      # accumulated JSON string from stream deltas
    tool_arguments: dict = None   # parsed after stream ends

    @property
    def is_pending(self) -> bool:
        return self.tool_name is not None

    def finalize(self):
        """Parse raw input string into dict. Call once after stream ends."""
        if self.tool_input_raw:
            self.tool_arguments = json.loads(self.tool_input_raw)
        return self


################################################################################
# SECTION: StreamMetrics + StreamResult Dataclasses
################################################################################

@dataclass
class StreamMetrics:
    input_tokens:  int = 0
    output_tokens: int = 0
    total_tokens:  int = 0
    latency_ms:    int = 0


@dataclass
class StreamResult:
    """Everything extracted from one converse_stream call."""
    text:            str = ""
    stop_reason:     str = ""
    tool_invocation: Optional[ToolInvocation] = None
    metrics:         StreamMetrics = field(default_factory=StreamMetrics)
    errors:          list = field(default_factory=list)

    @property
    def has_tool_call(self) -> bool:
        return (
            self.stop_reason == "tool_use"
            and self.tool_invocation is not None
            and self.tool_invocation.is_pending
        )

    @property
    def stop_reason_display(self) -> Optional[str]:
        """Human-readable stop reason, or None if normal end_turn / tool_use."""
        if self.stop_reason in ("end_turn", "tool_use"):
            return None
        return STOP_REASON_MESSAGES.get(self.stop_reason, self.stop_reason)


################################################################################
# SECTION: StreamProcessor  (pure EventStream parser — no UI dependencies)
################################################################################

def _handle_content_block_start(block_start: dict, tool_invocation: ToolInvocation):
    start = block_start.get('start', {})
    if 'toolUse' in start:
        tu = start['toolUse']
        tool_invocation.tool_use_id = tu['toolUseId']
        tool_invocation.tool_name   = tu['name']
        logger.info("Tool call started: id=%s name=%s", tu['toolUseId'], tu['name'])


def _handle_content_block_delta(
    delta: dict,
    result: StreamResult,
    tool_invocation: ToolInvocation,
    on_text_delta: Optional[Callable[[str], None]],
):
    if 'text' in delta:
        chunk = delta['text']
        result.text += chunk
        if on_text_delta:
            on_text_delta(chunk)

    elif 'toolUse' in delta:
        tool_invocation.tool_input_raw += delta['toolUse'].get('input', '')

    elif 'reasoningContent' in delta:
        rc = delta['reasoningContent']
        if 'text' in rc:
            logger.debug("Reasoning delta: %s", rc['text'])


def _handle_metadata(metadata: dict, result: StreamResult):
    if 'usage' in metadata:
        u = metadata['usage']
        result.metrics.input_tokens  = u.get('inputTokens', 0)
        result.metrics.output_tokens = u.get('outputTokens', 0)
        result.metrics.total_tokens  = u.get('totalTokens', 0)
    if 'metrics' in metadata:
        result.metrics.latency_ms = metadata['metrics'].get('latencyMs', 0)


def process_stream(
    stream,
    on_text_delta: Optional[Callable[[str], None]] = None,
) -> StreamResult:
    """
    Consume a converse_stream EventStream and return a StreamResult.

    Args:
        stream:        The EventStream from response['stream'].
        on_text_delta: Optional callback(str) fired for each text chunk.
                       Use this hook to update the UI incrementally.
    Returns:
        StreamResult with all extracted data.
    """
    result          = StreamResult()
    tool_invocation = ToolInvocation()

    for event in stream:

        if 'messageStart' in event:
            logger.debug("messageStart: role=%s", event['messageStart'].get('role'))

        elif 'contentBlockStart' in event:
            _handle_content_block_start(event['contentBlockStart'], tool_invocation)

        elif 'contentBlockDelta' in event:
            _handle_content_block_delta(
                event['contentBlockDelta']['delta'],
                result,
                tool_invocation,
                on_text_delta,
            )

        elif 'contentBlockStop' in event:
            pass  # reserved for future use

        elif 'messageStop' in event:
            result.stop_reason = event['messageStop'].get('stopReason', '')
            if result.stop_reason == 'tool_use':
                tool_invocation.finalize()
                result.tool_invocation = tool_invocation

        elif 'metadata' in event:
            _handle_metadata(event['metadata'], result)

        else:
            for exc_key in _EXCEPTION_EVENTS:
                if exc_key in event:
                    msg = event[exc_key].get('message', exc_key)
                    logger.error("Stream exception [%s]: %s", exc_key, msg)
                    result.errors.append(f"[{exc_key}] {msg}")

    return result


################################################################################
# SECTION: ToolRegistry
################################################################################

class ToolRegistry:
    """
    Registers tools and provides O(1) lookup + dispatch.

    Usage:
        registry   = ToolRegistry([calc_tool, wiki_tool, ...])
        tool_cfg   = registry.tool_config          # pass to converse_stream
        result     = registry.invoke(name, args)   # execute a tool
    """

    def __init__(self, tools: list):
        # keyed by toolSpec.name for O(1) lookup
        self._tools = {
            tool.definition['toolSpec']['name']: tool
            for tool in tools
        }

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def tool_config(self) -> dict:
        """Returns toolConfig dict ready for converse_stream."""
        return {"tools": [t.definition for t in self._tools.values()]}

    @property
    def tool_names(self) -> list:
        return list(self._tools.keys())

    def build_tool_summary(self) -> str | None:
        """
        Loop through all tools, collect non-null summaries,
        return formatted string or None if no summaries exist.
        """
        lines = [
            tool.summary()
            for tool in self._tools.values()
            if tool.summary() is not None      # ← skip if not overridden
        ]

        if not lines:
            return None

        return (
            "AVAILABLE TOOLS:\n"
            + "\n".join(f"  - {line}" for line in lines)
        )
    
    # ── Dispatch ──────────────────────────────────────────────────────────────

    def invoke(self, tool_name: str, tool_args: dict) -> Any:
        """
        Dispatch to the matching tool.
        Raises KeyError if tool_name is not registered.
        """
        if tool_name not in self._tools:
            raise KeyError(
                f"Unknown tool: '{tool_name}'. Available: {self.tool_names}"
            )
        tool = self._tools[tool_name]
        logger.info("Invoking tool '%s' with args: %s", tool_name, tool_args)
        return tool.invoke(tool_args.get('expression'), tool_args=tool_args)

    # ── Message builders ──────────────────────────────────────────────────────

    def build_tool_request_message(self, tool_invocation: ToolInvocation) -> dict:
        """Assistant-role message that records the tool call."""
        return {
            "role": "assistant",
            "content": [{
                "toolUse": {
                    "toolUseId": tool_invocation.tool_use_id,
                    "name":      tool_invocation.tool_name,
                    "input":     tool_invocation.tool_arguments,
                }
            }]
        }

    def build_tool_result_message(
        self,
        tool_invocation: ToolInvocation,
        result: Any,
    ) -> dict:
        """User-role message that delivers the tool result back to the model."""
        return {
            "role": "user",
            "content": [{
                "toolResult": {
                    "toolUseId": tool_invocation.tool_use_id,
                    "content":   [{"json": {"result": result}}],
                }
            }]
        }


################################################################################
# SECTION: ConversationManager  (single place where converse_stream is called)
################################################################################

class ConversationManager:
    """
    Orchestrates multi-turn conversations including tool-use loops.
    converse_stream is called ONLY inside _call_llm().

    The UI layer supplies callbacks so this class stays UI-agnostic:
        on_text_delta(chunk)              → each streamed text chunk
        on_stream_result(StreamResult)    → after each complete LLM response
        on_tool_invoked(name, args, res)  → after each tool execution
    """

    def __init__(
        self,
        bedrock_client,
        tool_registry:           ToolRegistry,
        model_id:                str,
        inference_config:        dict,
        system_prompts:          list,
        additional_model_fields: Optional[dict] = None,
    ):
        self.client                  = bedrock_client
        self.registry                = tool_registry
        self.model_id                = model_id
        self.inference_config        = inference_config
        self.system_prompts          = system_prompts
        self.additional_model_fields = additional_model_fields

    # ── Public API ────────────────────────────────────────────────────────────

    def run(
        self,
        message_history:  list,
        on_text_delta:    Optional[Callable[[str], None]] = None,
        on_stream_result: Optional[Callable] = None,
        on_tool_invoked:  Optional[Callable] = None,
    ) -> StreamResult:
        """
        Execute one user turn, resolving any tool-use loops automatically.
        Returns the final StreamResult after all tool calls are resolved.
        """
        messages = message_history.copy()

        while True:
            result = self._call_llm(messages, on_text_delta)

            if on_stream_result:
                on_stream_result(result)

            if not result.has_tool_call:
                return result                   # ← normal end_turn or error

            # ── Tool-use loop ─────────────────────────────────────────────────
            tool_inv    = result.tool_invocation
            tool_result = self.registry.invoke(tool_inv.tool_name,
                                               tool_inv.tool_arguments)

            if on_tool_invoked:
                on_tool_invoked(tool_inv.tool_name,
                                tool_inv.tool_arguments,
                                tool_result)

            messages.append(self.registry.build_tool_request_message(tool_inv))
            messages.append(self.registry.build_tool_result_message(tool_inv, tool_result))
            # loop → model sees tool result and continues

    # ── Private ───────────────────────────────────────────────────────────────

    def _call_llm(
        self,
        messages:      list,
        on_text_delta: Optional[Callable[[str], None]],
    ) -> StreamResult:
        """Single converse_stream call. Every LLM call goes through here."""
        try:
            kwargs = dict(
                modelId=self.model_id,
                messages=messages,
                system=self.system_prompts,
                toolConfig=self.registry.tool_config,
                inferenceConfig=self.inference_config,
            )
            if self.additional_model_fields:
                kwargs['additionalModelRequestFields'] = self.additional_model_fields

            response = self.client.converse_stream(**kwargs)
            return process_stream(response['stream'], on_text_delta)

        except ClientError as err:
            msg = err.response["Error"]["Message"]
            logger.error("ClientError in _call_llm: %s", msg)
            r = StreamResult()
            r.errors.append(msg)
            return r


################################################################################
# SECTION: File / Media Utilities
################################################################################

mime_mapping_image = {
    "image/png":  "png",
    "image/jpeg": "jpeg",
    "image/jpg":  "jpeg",
    "image/gif":  "gif",
    "image/webp": "webp",
}

mime_mapping_document = {
    "text/plain":                "txt",
    "application/vnd.ms-excel":  "csv",
    "application/pdf":           "pdf",
    "text/markdown":             "md",
}


def image_to_base64(image: Image.Image, fmt: str) -> str:
    buf = io.BytesIO()
    image.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


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


def process_uploaded_file(uploaded_file):
    """
    Read an uploaded Streamlit file and return
    (bytes, file_key, file_type, preview_widget_fn).

    preview_widget_fn is a zero-arg callable that renders a Streamlit preview,
    or None if no preview is needed.
    """
    if uploaded_file is None:
        return None, None, None, None

    file_type = uploaded_file.type
    file_key  = (uploaded_file.name
                 .replace(".", "_")
                 .replace(" ", "_"))

    if file_type in mime_mapping_image:
        raw   = uploaded_file.read()
        image = Image.open(io.BytesIO(raw))
        b64   = image_to_base64(image, mime_mapping_image[file_type].upper())
        preview = lambda: st.image(image, caption="Uploaded image",
                                   use_column_width=True)
        return raw, file_key, file_type, preview

    if file_type in mime_mapping_document:
        fmt = mime_mapping_document[file_type]

        if fmt == "csv":
            raw = base64.b64encode(uploaded_file.read())
            uploaded_file.seek(0)
            try:
                df = pd.read_csv(uploaded_file, encoding="utf-8")
                preview = lambda: st.dataframe(df)
            except Exception as e:
                preview = lambda: st.warning(f"CSV preview failed: {e}")
            return raw, file_key, file_type, preview

        if fmt == "pdf":
            raw = uploaded_file.read()
            preview = lambda: st.markdown(f"📄 **{uploaded_file.name}**")
            return raw, file_key, file_type, preview

        if fmt in ("txt", "md"):
            raw = base64.b64encode(uploaded_file.read())
            preview = lambda: st.markdown(f"📝 **{uploaded_file.name}**")
            return raw, file_key, file_type, preview

    st.warning(f"Unsupported file type: {file_type}")
    return None, None, None, None


################################################################################
# SECTION: Shared Streamlit Resources  (cached — created once per session)
################################################################################

@st.cache_resource
def get_bedrock_client():
    return boto3.client('bedrock-runtime', region_name=AWS_REGION)


@st.cache_resource
def get_tool_registry():
    return ToolRegistry([
        CalculatorBedrockConverseTool(),
        AcronymBedrockConverseTool(),
        UrlContentBedrockConverseTool(),
        WikipediaBedrockConverseTool(),
        DateTimeBedrockConverseTool(),
        SalesBedrockConverseTool(),
        ProductBedrockConverseTool(),
        ChartBedrockConverseTool(),
    ])


bedrock_client = get_bedrock_client()
tool_registry  = get_tool_registry()

tool_summary = tool_registry.build_tool_summary()


################################################################################
# SECTION: Streamlit Sidebar / Options
################################################################################

opt_model_id_list = [
    "global.anthropic.claude-sonnet-4-5-20250929-v1:0",
    "global.anthropic.claude-sonnet-4-20250514-v1:0",
    "cohere.command-r-v1:0",
    "cohere.command-r-plus-v1:0",
    "meta.llama3-8b-instruct-v1:0",
    "meta.llama3-70b-instruct-v1:0",
    "mistral.mistral-large-2402-v1:0",
]

# OPT_SYSTEM_MSG_DEFAULT = """You are a BI analyst assistant with access to sales data tools.

# When asked about sales performance, year-over-year comparisons, or trends:
# 1. ALWAYS fetch data for BOTH years using the sales_data tool
# 2. Compare month-by-month to identify where gaps occurred
# 3. Calculate percentage changes: ((current - previous) / previous * 100)
# 4. Identify the specific months where performance diverged
# 5. Look at profit_margin_pct alongside revenue — revenue can grow while margins shrink
# 6. Provide a structured response:
#    - Overall YoY summary (total revenue, growth %)
#    - Monthly trend analysis (which months underperformed and by how much)
#    - Root cause identification (is it volume/units or pricing/margin?)
#    - Actionable insights

# Be specific with numbers. Always cite the data returned by the tool.


# IMPORTANT: Call tools ONE AT A TIME. 
# Do not make parallel tool calls.
# Wait for each tool result before calling the next tool.
# """

# ── System Prompt ─────────────────────────────────────────────────────────────
def build_default_system_prompt(registry: ToolRegistry) -> str:
    tool_summary = registry.build_tool_summary()
    return "\n\n".join(filter(None, [
        "You are a BI analyst assistant.",
        tool_summary,
        "Call tools ONE AT A TIME. Wait for each result before calling the next.",
    ]))

OPT_SYSTEM_MSG_DEFAULT = build_default_system_prompt(tool_registry)

with st.sidebar:
    opt_model_id    = st.selectbox("Model ID", opt_model_id_list, index=0,
                                   key="model_id")
    opt_temperature = st.slider("Temperature", 0.0, 1.0, 0.1, 0.1,
                                key="temperature")
    opt_max_tokens  = st.slider("Max Tokens", 0, 4096, 2048, 1,
                                key="max_tokens")
    opt_system_msg  = st.text_area("System Message",
                                   OPT_SYSTEM_MSG_DEFAULT,
                                   key="system_msg")
    opt_show_metrics = st.checkbox("Show Invocation Metrics", value=False,
                                   key="show_metrics")


################################################################################
# SECTION: Streamlit Page Setup + Session State
################################################################################

with st.container(horizontal=True, vertical_alignment="center"):
    st.markdown("💬 Converse Tool")
    show_examples = st.toggle("Examples", value=False)
st.markdown(f"Tools: {', '.join(tool_registry.tool_names)}")

if show_examples:
    st.info("""
**💡 Example Questions to Try**

**Year-over-Year Analysis**
- Compare 2024 vs 2023 sales performance. What caused any underperformance?
- Which months in 2024 were worse than 2023 and by how much?
- Was the 2024 dip a volume problem or a margin problem?

**Monthly Drill-Down**
- Show me June 2024 sales breakdown by region and category
- What was the best performing month in 2023?
- How did Q4 2024 compare to Q4 2023?

**Regional Analysis**
- Which region performed better in 2024?
- Which region recovered faster in Q3 2024?
- Compare North vs South region for the full year 2024

**General**
- What is the total revenue for 2024?
- Which category has better profit margins?

**Product Catalog (NL-to-SQL)**
- Show me all products under $100
- Which products come in red?
- What is the most expensive product in each category?
- List all color options for the Laptop Pro
- Which product has the highest rating?
- How many color variants does each product have?
- Show products launched in 2023 with rating above 4.5
- Which color has the most stock across all products?
            """)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

st.markdown(f"{len(st.session_state.messages)}/{MAX_MESSAGES}")


################################################################################
# SECTION: Render Message History
################################################################################

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        contents = msg["content"]
        text     = contents[0].get("text", "")

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


################################################################################
# SECTION: File Uploader Widget
################################################################################

uploaded_file = st.file_uploader(
    "Attach file",
    type=["PNG", "JPEG", "TXT", "CSV", "PDF", "MD"],
    accept_multiple_files=False,
    label_visibility="collapsed",
    key=f"uploader_{st.session_state.uploader_key}",
)

file_bytes, file_key, file_type, file_preview = process_uploaded_file(uploaded_file)
if file_preview:
    file_preview()


################################################################################
# SECTION: Chat Input + Conversation Execution
################################################################################

prompt = st.chat_input()

if prompt:
    # ── Build user message ────────────────────────────────────────────────────
    user_message = build_user_message(
        prompt=prompt,
        uploaded_file=uploaded_file,
        uploaded_file_bytes=file_bytes,
        uploaded_file_type=file_type,
        uploaded_file_key=file_key,
    )

    message_history = st.session_state.messages.copy()
    message_history.append(user_message)

    st.chat_message("user").write(prompt)

    # ── UI state for streaming callbacks ──────────────────────────────────────
    accumulated = {"text": ""}

    with st.chat_message("assistant"):
        result_area      = st.empty()
        result_container = st.container(border=True)

        # ── Callbacks ─────────────────────────────────────────────────────────

        def on_text_delta(chunk: str):
            """Fired for every streamed text chunk."""
            accumulated["text"] += chunk
            result_area.markdown(accumulated["text"])

        def on_stream_result(result: StreamResult):
            """Fired after each complete LLM response."""
            if result.stop_reason_display:
                result_area.markdown(
                    f"{accumulated['text']}\n\n"
                    f":red[Generation Stopped: {result.stop_reason_display}]"
                )
            for err in result.errors:
                st.error(err)
            if opt_show_metrics:
                m = result.metrics
                result_container.write(
                    f"tokens in={m.input_tokens} out={m.output_tokens} "
                    f"total={m.total_tokens} latency={m.latency_ms}ms"
                )

        ##
        def on_tool_invoked_render_chart(tool_args: dict, tool_result: Any):
            """Renders chart using config from tool_args, data from tool_result."""

            data  = tool_result.get("data", [])
            x     = tool_args["x_label"]
            y     = tool_args["y_label"]
            title = tool_args["title"]
            color = tool_args.get("color", "#4A90D9")
            ctype = tool_args["chart_type"]

            if not data:
                st.warning("Chart error: no data in tool result.")
                return

            df = pd.DataFrame(data)

            if x not in df.columns or y not in df.columns:
                st.warning(f"Chart error: columns '{x}' or '{y}' not found.")
                return

            with result_container:
                st.markdown(f"**{title}**")
                chart_df = df.set_index(x)[[y]]
                if ctype == "bar":
                    st.bar_chart(chart_df, color=color)
                elif ctype == "line":
                    st.line_chart(chart_df, color=color)
                elif ctype == "area":
                    st.area_chart(chart_df, color=color)


        def on_tool_invoked_render_part(tool_name: str, tool_args: dict, tool_result: Any):
            """Tool-name based renderer."""

            if tool_name == "product_query":
                with result_container:
                    st.markdown(":blue[🗄️ **Generated SQL:**]")
                    st.code(
                        tool_args["sql"],
                        language="sql",
                        wrap_lines=True,
                    )

            elif tool_name == "render_chart":
                on_tool_invoked_render_chart(tool_args, tool_result)
        ##
        def on_tool_invoked(tool_name: str, tool_args: dict, tool_result: Any):
            """Fired after each tool execution."""
            accumulated["text"] += (
                f"\n\n:blue[🔧 **Tool:** `{tool_name}`]\n"
                f"```json\n{json.dumps(tool_args, indent=2)}\n```\n"
                f"**Result:** `{tool_result}`\n\n"
            )

            result_area.markdown(accumulated["text"])

            on_tool_invoked_render_part(tool_name, tool_args, tool_result)

        # ── Run ───────────────────────────────────────────────────────────────
        manager = ConversationManager(
            bedrock_client=bedrock_client,
            tool_registry=tool_registry,
            model_id=opt_model_id,
            inference_config={
                "temperature": opt_temperature,
                "maxTokens":   opt_max_tokens,
            },
            system_prompts=[{"text": opt_system_msg}],
        )

        with st.spinner("Processing..."):
            final_result = manager.run(
                message_history,
                on_text_delta=on_text_delta,
                on_stream_result=on_stream_result,
                on_tool_invoked=on_tool_invoked,
            )

    # ── Persist to session state ──────────────────────────────────────────────
    assistant_message = {
        "role":    "assistant",
        "content": [{"text": accumulated["text"]}],
    }
    st.session_state.messages.append(user_message)
    st.session_state.messages.append(assistant_message)

    # Trim history to MAX_MESSAGES
    msgs = st.session_state.messages
    if len(msgs) > MAX_MESSAGES:
        del msgs[0 : len(msgs) - MAX_MESSAGES]

    # Reset uploader after successful submission
    if uploaded_file:
        st.session_state.uploader_key += 1
        st.rerun()