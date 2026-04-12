import streamlit as st
import boto3
import json
import logging
import cmn_settings
from PIL import Image
import io
import base64
import pandas as pd

from cmn.view.mime_constants import mime_mapping_image, mime_mapping_document
from cmn.view import CONVERSE_TOOL_GUIDE
from cmn.view.processor.file_uploader_chat import render_file_uploader

from cmn.bedrock.converse import ConversationManager, StreamResult
from cmn.bedrock.client_manager import BedrockClientFactory

#from cmn.bedrock_converse_tools import CalculatorBedrockConverseTool
#from cmn.bedrock_converse_tools_acronym import AcronymBedrockConverseTool
#from cmn.bedrock_converse_tools_url import UrlContentBedrockConverseTool
#from cmn.bedrock_converse_tools_wikipedia import WikipediaBedrockConverseTool
#from cmn.bedrock_converse_tools_datetime import DateTimeBedrockConverseTool
#from cmn.bedrock_converse_tools_sales import SalesBedrockConverseTool
#from cmn.bedrock_converse_tools_product import ProductBedrockConverseTool
#from cmn.bedrock_converse_tools_chart import ChartBedrockConverseTool
#from cmn.bedrock_converse_tools_sales_kpi import SalesKpiBedrockConverseTool
#from cmn.bedrock_converse_tools_sales_anomaly import SalesAnomalyBedrockConverseTool
#from cmn.bedrock_converse_tools_sales_forecast import SalesForecastBedrockConverseTool

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
#from cmn.tools.tool import DateTimeBedrockConverseTool, HolidayBedrockConverseTool
#from cmn.tools.tool import AwsDocsBedrockConverseTool
from cmn.tools.tool import EDAProfileBedrockConverseTool
from cmn.tools.tool import EDACorrelationBedrockConverseTool
from cmn.tools.tool import EDAGroupBedrockConverseTool
#from cmn.tools.tool import CalculatorBedrockConverseTool
#from cmn.tools.tool import PptxBedrockConverseTool
#from cmn.tools.tool import PdfBedrockConverseTool


AWS_REGION = cmn_settings.AWS_REGION
MAX_MESSAGES = 100 * 2

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

################################################################################
# SECTION: ToolInvocation Dataclass
################################################################################

from dataclasses import dataclass, field
from typing import Any, Optional, Callable


@dataclass
class InvocationStat:
    input_tokens:  int = 0
    output_tokens: int = 0
    total_tokens:  int = 0
    latency_ms:    int = 0
    llm_calls:     int = 0
    tools_called:  list = field(default_factory=list)

    def accumulate(self, result: StreamResult) -> None:
        m = result.metrics
        self.input_tokens  += m.input_tokens
        self.output_tokens += m.output_tokens
        self.total_tokens  += m.total_tokens
        self.latency_ms    += m.latency_ms
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

# STOP_REASON_MESSAGES = {
#     "max_tokens":           "Insufficient Tokens. Increase MaxToken Settings.",
#     "guardrail_intervened": "Response blocked by guardrail.",
#     "content_filtered":     "Content was filtered.",
# }

# @dataclass
# class ToolInvocation:
#     """Captures a single tool call requested by the model."""
#     tool_use_id:    str = None
#     tool_name:      str = None
#     tool_input_raw: str = ""      # accumulated JSON string from stream deltas
#     tool_arguments: dict = None   # parsed after stream ends

#     @property
#     def is_pending(self) -> bool:
#         return self.tool_name is not None

#     def finalize(self):
#         """Parse raw input string into dict. Call once after stream ends."""
#         if self.tool_input_raw:
#             self.tool_arguments = json.loads(self.tool_input_raw)
#         else:
#             self.tool_arguments = {}    # ← empty dict instead of None
#         return self


# ################################################################################
# # SECTION: StreamMetrics + StreamResult Dataclasses
# ################################################################################

# @dataclass
# class StreamMetrics:
#     input_tokens:  int = 0
#     output_tokens: int = 0
#     total_tokens:  int = 0
#     latency_ms:    int = 0


# @dataclass
# class StreamResult:
#     """Everything extracted from one converse_stream call."""
#     text:             str = ""
#     stop_reason:      str = ""
#     tool_invocation:  Optional[ToolInvocation] = None       # first tool (backward compat)
#     tool_invocations: list = field(default_factory=list)    # ← all tools
#     metrics:          StreamMetrics = field(default_factory=StreamMetrics)
#     errors:           list = field(default_factory=list)

#     @property
#     def has_tool_call(self) -> bool:
#         return (
#             self.stop_reason == "tool_use"
#             and bool(self.tool_invocations)
#         )

#     @property
#     def stop_reason_display(self) -> Optional[str]:
#         """Human-readable stop reason, or None if normal end_turn / tool_use."""
#         if self.stop_reason in ("end_turn", "tool_use"):
#             return None
#         return STOP_REASON_MESSAGES.get(self.stop_reason, self.stop_reason)


################################################################################
# SECTION: StreamProcessor  (pure EventStream parser — no UI dependencies)
################################################################################

# def _handle_content_block_start(block_start: dict, tool_invocation: ToolInvocation):
#     start = block_start.get('start', {})
#     if 'toolUse' in start:
#         tu = start['toolUse']
#         tool_invocation.tool_use_id = tu['toolUseId']
#         tool_invocation.tool_name   = tu['name']
#         logger.info("Tool call started: id=%s name=%s", tu['toolUseId'], tu['name'])


# def _handle_content_block_delta(
#     delta: dict,
#     result: StreamResult,
#     tool_invocation: ToolInvocation,
#     on_text_delta: Optional[Callable[[str], None]],
# ):
#     if 'text' in delta:
#         chunk = delta['text']
#         result.text += chunk
#         if on_text_delta:
#             on_text_delta(chunk)

#     elif 'toolUse' in delta:
#         tool_invocation.tool_input_raw += delta['toolUse'].get('input', '')

#     elif 'reasoningContent' in delta:
#         rc = delta['reasoningContent']
#         if 'text' in rc:
#             logger.debug("Reasoning delta: %s", rc['text'])


# def _handle_metadata(metadata: dict, result: StreamResult):
#     if 'usage' in metadata:
#         u = metadata['usage']
#         result.metrics.input_tokens  = u.get('inputTokens', 0)
#         result.metrics.output_tokens = u.get('outputTokens', 0)
#         result.metrics.total_tokens  = u.get('totalTokens', 0)
#     if 'metrics' in metadata:
#         result.metrics.latency_ms = metadata['metrics'].get('latencyMs', 0)


# def process_stream(
#     stream,
#     on_text_delta: Optional[Callable[[str], None]] = None,
# ) -> StreamResult:

#     result           = StreamResult()
#     tool_invocations = []       # ← collect all tool calls
#     current_tool     = None     # ← track active streaming tool

#     for event in stream:

#         if 'messageStart' in event:
#             logger.debug("messageStart: role=%s", event['messageStart'].get('role'))

#         elif 'contentBlockStart' in event:
#             start = event['contentBlockStart'].get('start', {})
#             if 'toolUse' in start:
#                 # ── New tool call — create fresh ToolInvocation ───────────
#                 tu           = start['toolUse']
#                 current_tool = ToolInvocation(
#                     tool_use_id = tu['toolUseId'],
#                     tool_name   = tu['name'],
#                 )
#                 tool_invocations.append(current_tool)
#                 logger.info("Tool call started: id=%s name=%s", tu['toolUseId'], tu['name'])
#             else:
#                 current_tool = None     # text block, not a tool

#         elif 'contentBlockDelta' in event:
#             delta = event['contentBlockDelta']['delta']

#             if 'text' in delta:
#                 chunk = delta['text']
#                 result.text += chunk
#                 if on_text_delta:
#                     on_text_delta(chunk)

#             elif 'toolUse' in delta:
#                 # ── Accumulate into CURRENT tool only ─────────────────────
#                 if current_tool is not None:
#                     current_tool.tool_input_raw += delta['toolUse'].get('input', '')

#             elif 'reasoningContent' in delta:
#                 rc = delta['reasoningContent']
#                 if 'text' in rc:
#                     logger.debug("Reasoning delta: %s", rc['text'])

#         elif 'contentBlockStop' in event:
#             # ── Block done — finalize current tool ────────────────────────
#             if current_tool is not None:
#                 current_tool.finalize()
#                 logger.info("Tool call finalized: name=%s args=%s",
#                             current_tool.tool_name,
#                             current_tool.tool_arguments)
#                 current_tool = None     # reset for next block

#         elif 'messageStop' in event:
#             result.stop_reason = event['messageStop'].get('stopReason', '')
#             if result.stop_reason == 'tool_use' and tool_invocations:
#                 result.tool_invocations = tool_invocations
#                 result.tool_invocation  = tool_invocations[0]  # backward compat

#         elif 'metadata' in event:
#             _handle_metadata(event['metadata'], result)

#         else:
#             for exc_key in _EXCEPTION_EVENTS:
#                 if exc_key in event:
#                     msg = event[exc_key].get('message', exc_key)
#                     logger.error("Stream exception [%s]: %s", exc_key, msg)
#                     result.errors.append(f"[{exc_key}] {msg}")

#     return result


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


renderer_registry = get_renderer_registry()


################################################################################
# SECTION: ConversationManager  (single place where converse_stream is called)
################################################################################

# class ConversationManager:
#     """
#     Orchestrates multi-turn conversations including tool-use loops.
#     converse_stream is called ONLY inside _call_llm().

#     The UI layer supplies callbacks so this class stays UI-agnostic:
#         on_text_delta(chunk)              → each streamed text chunk
#         on_stream_result(StreamResult)    → after each complete LLM response
#         on_tool_invoked(name, args, res)  → after each tool execution
#     """

#     def __init__(
#         self,
#         bedrock_client,
#         tool_registry:           ToolRegistry,
#         model_id:                str,
#         inference_config:        dict,
#         system_prompts:          list,
#         additional_model_fields: Optional[dict] = None,
#     ):
#         self.client                  = bedrock_client
#         self.registry                = tool_registry
#         self.model_id                = model_id
#         self.inference_config        = inference_config
#         self.system_prompts          = system_prompts
#         self.additional_model_fields = additional_model_fields

#     # ── Public API ────────────────────────────────────────────────────────────

#     def run(
#         self,
#         message_history:  list,
#         on_text_delta:    Optional[Callable[[str], None]] = None,
#         on_stream_result: Optional[Callable] = None,
#         on_tool_invoked:  Optional[Callable] = None,
#     ) -> StreamResult:

#         messages = message_history.copy()

#         while True:
#             result = self._call_llm(messages, on_text_delta)

#             if on_stream_result:
#                 on_stream_result(result)

#             if not result.has_tool_call:
#                 return result

#             # ── Build single assistant message for ALL tool calls ─────────────
#             assistant_content = [
#                 {
#                     "toolUse": {
#                         "toolUseId": tool_inv.tool_use_id,
#                         "name":      tool_inv.tool_name,
#                         "input":     tool_inv.tool_arguments,
#                     }
#                 }
#                 for tool_inv in result.tool_invocations
#             ]
#             messages.append({"role": "assistant", "content": assistant_content})

#             # ── Execute each tool + build ONE user message with all results ────
#             tool_results_content = []

#             for tool_inv in result.tool_invocations:
#                 tool_result = self.registry.invoke(
#                     tool_inv.tool_name,
#                     tool_inv.tool_arguments,
#                 )

#                 if on_tool_invoked:
#                     on_tool_invoked(tool_inv.tool_name,
#                                     tool_inv.tool_arguments,
#                                     tool_result)

#                 tool_results_content.append({
#                     "toolResult": {
#                         "toolUseId": tool_inv.tool_use_id,
#                         "content":   [{"json": {"result": tool_result}}],
#                     }
#                 })

#             messages.append({"role": "user", "content": tool_results_content})
#             # loop → model sees all tool results and continues

#     # ── Private ───────────────────────────────────────────────────────────────

#     def _call_llm(
#         self,
#         messages:      list,
#         on_text_delta: Optional[Callable[[str], None]],
#     ) -> StreamResult:
#         """Single converse_stream call. Every LLM call goes through here."""
#         try:
#             kwargs = dict(
#                 modelId=self.model_id,
#                 messages=messages,
#                 system=self.system_prompts,
#                 toolConfig=self.registry.tool_config,
#                 inferenceConfig=self.inference_config,
#             )
#             if self.additional_model_fields:
#                 kwargs['additionalModelRequestFields'] = self.additional_model_fields

#             response = self.client.converse_stream(**kwargs)
#             return process_stream(response['stream'], on_text_delta)

#         except ClientError as err:
#             msg = err.response["Error"]["Message"]
#             logger.error("ClientError in _call_llm: %s", msg)
#             r = StreamResult()
#             r.errors.append(msg)
#             return r


################################################################################
# SECTION: File / Media Utilities
################################################################################

# mime_mapping_image = {
#     "image/png":  "png",
#     "image/jpeg": "jpeg",
#     "image/jpg":  "jpeg",
#     "image/gif":  "gif",
#     "image/webp": "webp",
# }

# mime_mapping_document = {
#     "text/plain":                "txt",
#     "application/vnd.ms-excel":  "csv",
#     "application/pdf":           "pdf",
#     "text/markdown":             "md",
# }


# def image_to_base64(image: Image.Image, fmt: str) -> str:
#     buf = io.BytesIO()
#     image.save(buf, format=fmt)
#     return base64.b64encode(buf.getvalue()).decode("utf-8")


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


# def process_uploaded_file(uploaded_file):
#     """
#     Read an uploaded Streamlit file and return
#     (bytes, file_key, file_type, preview_widget_fn).

#     preview_widget_fn is a zero-arg callable that renders a Streamlit preview,
#     or None if no preview is needed.
#     """
#     if uploaded_file is None:
#         return None, None, None, None

#     file_type = uploaded_file.type
#     file_key  = (uploaded_file.name
#                  .replace(".", "_")
#                  .replace(" ", "_"))

#     if file_type in mime_mapping_image:
#         raw   = uploaded_file.read()
#         image = Image.open(io.BytesIO(raw))
#         b64   = image_to_base64(image, mime_mapping_image[file_type].upper())
#         preview = lambda: st.image(image, caption="Uploaded image",
#                                    use_column_width=True)
#         return raw, file_key, file_type, preview

#     if file_type in mime_mapping_document:
#         fmt = mime_mapping_document[file_type]

#         if fmt == "csv":
#             raw = base64.b64encode(uploaded_file.read())
#             uploaded_file.seek(0)
#             try:
#                 df = pd.read_csv(uploaded_file, encoding="utf-8")
#                 preview = lambda: st.dataframe(df)
#             except Exception as e:
#                 preview = lambda: st.warning(f"CSV preview failed: {e}")
#             return raw, file_key, file_type, preview

#         if fmt == "pdf":
#             raw = uploaded_file.read()
#             preview = lambda: st.markdown(f"📄 **{uploaded_file.name}**")
#             return raw, file_key, file_type, preview

#         if fmt in ("txt", "md"):
#             raw = base64.b64encode(uploaded_file.read())
#             preview = lambda: st.markdown(f"📝 **{uploaded_file.name}**")
#             return raw, file_key, file_type, preview

#     st.warning(f"Unsupported file type: {file_type}")
#     return None, None, None, None


################################################################################
# SECTION: Shared Streamlit Resources  (cached — created once per session)
################################################################################

@st.cache_resource
def get_bedrock_client():
    return BedrockClientFactory.bedrock_runtime(region=AWS_REGION)


@st.cache_resource
def get_tool_registry():
    return ToolRegistry([
        CalculatorBedrockConverseTool(),
        AcronymBedrockConverseTool(),
        UrlContentBedrockConverseTool(),
        WikipediaBedrockConverseTool(),
        AwsDocsBedrockConverseTool(),
        DateTimeBedrockConverseTool(),
        SalesBedrockConverseTool(),
        ProductBedrockConverseTool(),
        ChartBedrockConverseTool(),
        SalesKpiBedrockConverseTool(),
        SalesForecastBedrockConverseTool(),
        SalesAnomalyBedrockConverseTool(),
        # HolidayBedrockConverseTool(),
        # EDAProfileBedrockConverseTool(),
        # EDACorrelationBedrockConverseTool(),
        # EDAGroupBedrockConverseTool(),
        PptxBedrockConverseTool(),
        PdfBedrockConverseTool(),
    ])


bedrock_client = get_bedrock_client()
tool_registry  = get_tool_registry()

#tool_summary = tool_registry.build_tool_summary()


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
    "moonshotai.kimi-k2.5",
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
    opt_model_id    = st.selectbox("Model ID", opt_model_id_list, index=0, key="model_id")
    opt_temperature = st.slider("Temperature", 0.0, 1.0, 0.1, 0.1, key="temperature")
    opt_max_tokens  = st.slider("Max Tokens", 0, 8192, 4096, 1, key="max_tokens")
    opt_system_msg  = st.text_area("System Message", OPT_SYSTEM_MSG_DEFAULT, key="system_msg")
    opt_show_metrics = st.checkbox("Show Invocation Metrics", value=False, key="show_metrics")

    with st.expander("Tools"):
        st.markdown(f"Tools: {', '.join(tool_registry.tool_names)}")

################################################################################
# SECTION: Streamlit Page Setup + Session State
################################################################################

with st.container(horizontal=True, vertical_alignment="center"):
    st.markdown("💬 Converse Tool")
    show_examples = st.toggle("Examples", value=False)
#st.markdown(f"Tools: {', '.join(tool_registry.tool_names)}")

if show_examples:
    st.info(CONVERSE_TOOL_GUIDE)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "invocation_stats" not in st.session_state:
    st.session_state.invocation_stats = []

if len(st.session_state.invocation_stats) != len(st.session_state.messages):
    st.session_state.invocation_stats = [None] * len(st.session_state.messages)

if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

st.markdown(f"{len(st.session_state.messages)}/{MAX_MESSAGES}")


################################################################################
# SECTION: Render Message History
################################################################################

#for msg in st.session_state.messages:
for idx, msg in enumerate(st.session_state.messages):
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

        if opt_show_metrics and msg["role"] == "assistant":
            stat = st.session_state.invocation_stats[idx]
            if stat is not None:
                st.caption(stat.as_markdown())

        #else:
        #    st.markdown(text)


################################################################################
# SECTION: File Uploader Widget
################################################################################

# uploaded_file = st.file_uploader(
#     "Attach file",
#     type=["PNG", "JPEG", "TXT", "CSV", "PDF", "MD"],
#     accept_multiple_files=False,
#     label_visibility="collapsed",
#     key=f"uploader_{st.session_state.uploader_key}",
# )

# file_bytes, file_key, file_type, file_preview = process_uploaded_file(uploaded_file)
# if file_preview:
#     file_preview()

uploaded_file, file_bytes, file_key, file_type, file_preview = render_file_uploader(
    st.session_state.uploader_key
)


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

    # ── Per-turn accumulators ─────────────────────────────────────────────────
    # accumulated["text"] grows with every streamed chunk and tool annotation.
    # turn_stat collects token/latency metrics across all LLM calls in this turn
    # (there may be multiple calls when tools are used in a loop).
    accumulated = {"text": ""}
    turn_stat   = InvocationStat()

    with st.chat_message("assistant"):
        # result_area  — st.empty() single-slot placeholder.
        #                Overwritten on every streamed text chunk so the user
        #                sees the response grow in place.
        result_area      = st.empty()
        # result_container — st.container() that accumulates widgets.
        #                    renderer_registry appends charts, tables and other
        #                    rich tool visualisations here without overwriting
        #                    previous content.
        result_container = st.container(border=False)

        # ── Callbacks ─────────────────────────────────────────────────────────

        def on_text_delta(chunk: str):
            """Fired for every streamed text chunk."""
            accumulated["text"] += chunk
            result_area.markdown(accumulated["text"])

        # def on_stream_result(result: StreamResult):
        #     """Fired after each complete LLM response."""
        #     if result.stop_reason_display:
        #         result_area.markdown(
        #             f"{accumulated['text']}\n\n"
        #             f":red[Generation Stopped: {result.stop_reason_display}]"
        #         )
        #     for err in result.errors:
        #         st.error(err)
        #     if opt_show_metrics:
        #         m = result.metrics
        #         result_container.write(
        #             f"tokens in={m.input_tokens} out={m.output_tokens} "
        #             f"total={m.total_tokens} latency={m.latency_ms}ms"
        #         )

        # ── Callback: fired once after each complete LLM response ─────────────
        # Called multiple times when tool loops cause additional LLM calls.
        # Accumulates metrics into turn_stat for later storage in session state.
        def on_stream_result(result: StreamResult):
            turn_stat.accumulate(result)
            if result.stop_reason_display:
                result_area.markdown(
                    f"{accumulated['text']}\n\n"
                    f":red[Generation Stopped: {result.stop_reason_display}]"
                )
            for err in result.errors:
                st.error(err)

        # ── Callback: fired after each tool is executed ───────────────────────
        # Appends a tool annotation to the streamed text and delegates rich
        # rendering (charts, tables etc.) to the renderer registry.
        def on_tool_invoked(tool_name: str, tool_args: dict, tool_result: Any):
            turn_stat.record_tool(tool_name)
            accumulated["text"] += (
                f"\n\n:blue[🔧 **Tool:** `{tool_name}`]\n"
                f"```json\n{json.dumps(tool_args, indent=2)}\n```\n"
                f"**Result:** `{tool_result}`\n\n"
            )
            result_area.markdown(accumulated["text"])
            renderer_registry.render(tool_name, tool_args, tool_result, result_container)

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

        with st.spinner("Processing...", show_time=True, width="content"):
            final_result = manager.run(
                message_history,
                on_text_delta=on_text_delta,
                on_stream_result=on_stream_result,
                on_tool_invoked=on_tool_invoked,
            )

        # ── Show live metrics for this turn directly below the response ───────
        # These are also stored in session state so they reappear on rerun.
        if opt_show_metrics:
            st.caption(turn_stat.as_markdown())

    # ── Persist to session state ──────────────────────────────────────────────
    assistant_message = {
        "role":    "assistant",
        "content": [{"text": accumulated["text"]}],
    }
    st.session_state.messages.append(user_message)
    st.session_state.invocation_stats.append(None)
    st.session_state.messages.append(assistant_message)
    st.session_state.invocation_stats.append(turn_stat)

    # Trim history to MAX_MESSAGES
    msgs = st.session_state.messages
    stats = st.session_state.invocation_stats
    if len(msgs) > MAX_MESSAGES:
        excess = len(msgs) - MAX_MESSAGES
        del msgs[0:excess]
        del stats[0:excess]

    # Reset uploader after successful submission
    if uploaded_file:
        st.session_state.uploader_key += 1
        st.rerun()