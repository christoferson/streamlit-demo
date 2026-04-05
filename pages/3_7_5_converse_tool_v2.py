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
import plotly.express as px
from botocore.exceptions import BotoCoreError, ClientError

from cmn.view.mime_constants import mime_mapping_image, mime_mapping_document

#from cmn.bedrock_converse_tools import CalculatorBedrockConverseTool
from cmn.bedrock_converse_tools_acronym import AcronymBedrockConverseTool
from cmn.bedrock_converse_tools_url import UrlContentBedrockConverseTool
#from cmn.bedrock_converse_tools_wikipedia import WikipediaBedrockConverseTool
#from cmn.bedrock_converse_tools_datetime import DateTimeBedrockConverseTool
from cmn.bedrock_converse_tools_sales import SalesBedrockConverseTool
from cmn.bedrock_converse_tools_product import ProductBedrockConverseTool
from cmn.bedrock_converse_tools_chart import ChartBedrockConverseTool
from cmn.bedrock_converse_tools_sales_kpi import SalesKpiBedrockConverseTool
from cmn.bedrock_converse_tools_sales_anomaly import SalesAnomalyBedrockConverseTool
from cmn.bedrock_converse_tools_sales_forecast import SalesForecastBedrockConverseTool

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
from cmn.tools.tool import DateTimeBedrockConverseTool, HolidayBedrockConverseTool
from cmn.tools.tool import AwsDocsBedrockConverseTool
from cmn.tools.tool import EDAProfileBedrockConverseTool
from cmn.tools.tool import EDACorrelationBedrockConverseTool
from cmn.tools.tool import EDAGroupBedrockConverseTool
from cmn.tools.tool import CalculatorBedrockConverseTool
from cmn.tools.tool import PptxBedrockConverseTool
from cmn.tools.tool import PdfBedrockConverseTool
from cmn.tools.tool import (
    WikipediaBedrockConverseTool
)

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
        else:
            self.tool_arguments = {}    # ← empty dict instead of None
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
    text:             str = ""
    stop_reason:      str = ""
    tool_invocation:  Optional[ToolInvocation] = None       # first tool (backward compat)
    tool_invocations: list = field(default_factory=list)    # ← all tools
    metrics:          StreamMetrics = field(default_factory=StreamMetrics)
    errors:           list = field(default_factory=list)

    @property
    def has_tool_call(self) -> bool:
        return (
            self.stop_reason == "tool_use"
            and bool(self.tool_invocations)
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

    result           = StreamResult()
    tool_invocations = []       # ← collect all tool calls
    current_tool     = None     # ← track active streaming tool

    for event in stream:

        if 'messageStart' in event:
            logger.debug("messageStart: role=%s", event['messageStart'].get('role'))

        elif 'contentBlockStart' in event:
            start = event['contentBlockStart'].get('start', {})
            if 'toolUse' in start:
                # ── New tool call — create fresh ToolInvocation ───────────
                tu           = start['toolUse']
                current_tool = ToolInvocation(
                    tool_use_id = tu['toolUseId'],
                    tool_name   = tu['name'],
                )
                tool_invocations.append(current_tool)
                logger.info("Tool call started: id=%s name=%s", tu['toolUseId'], tu['name'])
            else:
                current_tool = None     # text block, not a tool

        elif 'contentBlockDelta' in event:
            delta = event['contentBlockDelta']['delta']

            if 'text' in delta:
                chunk = delta['text']
                result.text += chunk
                if on_text_delta:
                    on_text_delta(chunk)

            elif 'toolUse' in delta:
                # ── Accumulate into CURRENT tool only ─────────────────────
                if current_tool is not None:
                    current_tool.tool_input_raw += delta['toolUse'].get('input', '')

            elif 'reasoningContent' in delta:
                rc = delta['reasoningContent']
                if 'text' in rc:
                    logger.debug("Reasoning delta: %s", rc['text'])

        elif 'contentBlockStop' in event:
            # ── Block done — finalize current tool ────────────────────────
            if current_tool is not None:
                current_tool.finalize()
                logger.info("Tool call finalized: name=%s args=%s",
                            current_tool.tool_name,
                            current_tool.tool_arguments)
                current_tool = None     # reset for next block

        elif 'messageStop' in event:
            result.stop_reason = event['messageStop'].get('stopReason', '')
            if result.stop_reason == 'tool_use' and tool_invocations:
                result.tool_invocations = tool_invocations
                result.tool_invocation  = tool_invocations[0]  # backward compat

        elif 'metadata' in event:
            _handle_metadata(event['metadata'], result)

        else:
            for exc_key in _EXCEPTION_EVENTS:
                if exc_key in event:
                    msg = event[exc_key].get('message', exc_key)
                    logger.error("Stream exception [%s]: %s", exc_key, msg)
                    result.errors.append(f"[{exc_key}] {msg}")

    return result


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

        messages = message_history.copy()

        while True:
            result = self._call_llm(messages, on_text_delta)

            if on_stream_result:
                on_stream_result(result)

            if not result.has_tool_call:
                return result

            # ── Build single assistant message for ALL tool calls ─────────────
            assistant_content = [
                {
                    "toolUse": {
                        "toolUseId": tool_inv.tool_use_id,
                        "name":      tool_inv.tool_name,
                        "input":     tool_inv.tool_arguments,
                    }
                }
                for tool_inv in result.tool_invocations
            ]
            messages.append({"role": "assistant", "content": assistant_content})

            # ── Execute each tool + build ONE user message with all results ────
            tool_results_content = []

            for tool_inv in result.tool_invocations:
                tool_result = self.registry.invoke(
                    tool_inv.tool_name,
                    tool_inv.tool_arguments,
                )

                if on_tool_invoked:
                    on_tool_invoked(tool_inv.tool_name,
                                    tool_inv.tool_arguments,
                                    tool_result)

                tool_results_content.append({
                    "toolResult": {
                        "toolUseId": tool_inv.tool_use_id,
                        "content":   [{"json": {"result": tool_result}}],
                    }
                })

            messages.append({"role": "user", "content": tool_results_content})
            # loop → model sees all tool results and continues

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
        SalesKpiBedrockConverseTool(),
        SalesAnomalyBedrockConverseTool(),
        SalesForecastBedrockConverseTool(),
        HolidayBedrockConverseTool(),
        AwsDocsBedrockConverseTool(),
        EDAProfileBedrockConverseTool(),
        EDACorrelationBedrockConverseTool(),
        EDAGroupBedrockConverseTool(),
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

**KPI Dashboard**
- Give me a KPI summary comparing 2024 vs 2023
- Show me the key metrics for 2024 sales performance
- Display a KPI dashboard for Q4 2024 vs Q4 2023

**Anomaly Detection**
- Find anomalies in 2024 revenue
- Which months had unusual sales in 2024?
- Detect anomalies in 2024 returns — are returns spiking anywhere?
- Find revenue anomalies in 2023 vs the yearly average

**Forecast**
- Forecast next 2 months revenue based on 2024 data
- Show 2024 monthly sales with 3 month forecast as a line chart
- Predict units sold for next 6 months using 2024 data
- What is the revenue trend forecast for early 2025?

** AWS Doc **
"What is the maximum size of an S3 object?"
"How does DynamoDB pricing work?"
"What are the Lambda concurrency limits?"
"How do I configure S3 lifecycle policies?"
"What is Amazon Bedrock and what models does it support?"

**EDA Profile**
- Profile 2024 sales data
- What does the 2024 sales dataset look like?
- Give me a summary of 2023 sales
            
**EDA**
- Profile 2024 sales data
- What correlates with revenue in 2024 sales?
- What drives gross profit in 2024?
- Show correlation between units sold and returns
            
**EDA**
- Compare 2024 revenue by region
- Which region has the highest average revenue in 2024?
- Compare units sold by category in 2023
- Is there a significant difference in revenue between regions?
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
        # def on_tool_invoked_render_chart(tool_args: dict, tool_result: Any):

        #     data  = tool_args.get("data") or tool_result.get("data", [])
        #     x     = tool_args["x_label"]
        #     y     = tool_args["y_label"]
        #     title = tool_args["title"]
        #     ctype = tool_args["chart_type"]

        #     if not data:
        #         with result_container:
        #             st.warning("Chart error: no data provided.")
        #         return

        #     df = pd.DataFrame(data)

        #     if x not in df.columns:
        #         with result_container:
        #             st.warning(f"Chart error: x='{x}' not found. Got: {list(df.columns)}")
        #         return

        #     # ── Detect long format — has a series/type/category column ───────────
        #     color_col = next(
        #         (c for c in ["series", "type", "category"] if c in df.columns),
        #         None
        #     )

        #     x_order = df[x].unique().tolist()   # preserve original order

        #     with result_container:
        #         st.markdown(f"**{title}**")

        #         if color_col:
        #             # ── Long format — color by series column ─────────────────────
        #             if ctype == "bar":
        #                 fig = px.bar(df, x=x, y=y, color=color_col, barmode="group")
        #             elif ctype == "line":
        #                 fig = px.line(df, x=x, y=y, color=color_col, markers=True)
        #             else:
        #                 fig = px.area(df, x=x, y=y, color=color_col)

        #         else:
        #             # ── Wide format — one column per series ───────────────────────
        #             chart_df  = df.set_index(x)
        #             y_columns = (
        #                 [y] if y in chart_df.columns
        #                 else chart_df.select_dtypes(include="number").columns.tolist()
        #             )
        #             plot_df = chart_df[y_columns].reset_index()

        #             if ctype == "bar":
        #                 fig = px.bar(plot_df, x=x, y=y_columns, barmode="group")
        #             elif ctype == "line":
        #                 fig = px.line(plot_df, x=x, y=y_columns, markers=True)
        #             else:
        #                 fig = px.area(plot_df, x=x, y=y_columns)

        #         fig.update_xaxes(categoryorder="array", categoryarray=x_order)
        #         st.plotly_chart(fig, width="content")
        
        # def on_tool_invoked_render_kpi(tool_args: dict, tool_result: Any):
        #     """Renders KPI cards from tool_result."""

        #     if tool_result.get("status") != "kpi_ready":
        #         return

        #     title   = tool_result.get("title", "")
        #     metrics = tool_result.get("metrics", [])

        #     if not metrics:
        #         return

        #     with result_container:
        #         if title:
        #             st.markdown(f"**{title}**")

        #         # ── Render in rows of 4 ───────────────────────────────────────────
        #         chunk_size = 4
        #         for i in range(0, len(metrics), chunk_size):
        #             row     = metrics[i : i + chunk_size]
        #             columns = st.columns(len(row))

        #             for col, metric in zip(columns, row):
        #                 with col:
        #                     st.metric(
        #                         label       = metric.get("label", ""),
        #                         value       = metric.get("value", ""),
        #                         delta       = metric.get("delta"),           # None if not provided
        #                         delta_color = metric.get("delta_color", "normal"),
        #                     )

        # def on_tool_invoked_render_anomaly(tool_args: dict, tool_result: Any):
        #     """Renders anomaly detection results."""

        #     anomalies = tool_result.get("anomalies", [])
        #     normal    = tool_result.get("normal",    [])
        #     metric    = tool_result.get("metric",    "revenue")
        #     year      = tool_result.get("year",      "")
        #     mean      = tool_result.get("mean",      0)

        #     if "error" in tool_result:
        #         with result_container:
        #             st.warning(f"Anomaly detection: {tool_result['error']}")
        #         return

        #     with result_container:
        #         st.markdown(f"**🔍 Anomaly Detection — {metric.title()} {year}**")
        #         st.caption(f"Yearly mean: {mean:,.0f} | Threshold: ±{tool_result.get('threshold', 1.5)} std dev")

        #         if not anomalies:
        #             st.success("✅ No anomalies detected — all months within normal range.")
        #             return

        #         # ── Anomaly cards ─────────────────────────────────────────────────
        #         cols = st.columns(len(anomalies)) if len(anomalies) <= 4 else st.columns(4)

        #         for i, entry in enumerate(anomalies):
        #             col = cols[i % 4] if len(anomalies) > 4 else cols[i]
        #             with col:
        #                 flag     = entry["flag"]
        #                 severity = entry.get("severity", "")
        #                 pct      = entry["pct_vs_mean"]
        #                 icon     = "🔴" if flag == "below_normal" else "🟢"

        #                 st.metric(
        #                     label       = f"{icon} {entry['month_name']}",
        #                     value       = f"{entry['value']:,.0f}",
        #                     delta       = f"{pct:+.1f}% vs mean",
        #                     delta_color = "normal" if flag == "above_normal" else "inverse",
        #                 )
        #                 st.caption(f"Z-score: {entry['zscore']} | {severity}")

        #         # ── Normal months summary ─────────────────────────────────────────
        #         if normal:
        #             normal_names = ", ".join(n["month_name"] for n in normal)
        #             st.caption(f"✅ Normal months: {normal_names}")

        # def on_tool_invoked_render_forecast(tool_args: dict, tool_result: Any):

        #     if "error" in tool_result:
        #         with result_container:
        #             st.warning(f"Forecast error: {tool_result['error']}")
        #         return

        #     actuals       = tool_result.get("actuals",  [])
        #     forecast      = tool_result.get("forecast", [])
        #     metric        = tool_result.get("metric",   "revenue")
        #     train_year    = tool_result.get("train_year")
        #     forecast_year = tool_result.get("forecast_year")
        #     model_info    = tool_result.get("model",    {})
        #     summary       = tool_result.get("summary",  {})

        #     with result_container:
        #         st.markdown(f"**📈 Sales Forecast — {metric.title()}**")

        #         # ── Model metrics ─────────────────────────────────────────────────
        #         col1, col2, col3 = st.columns(3)
        #         col1.metric("Trend",      summary.get("trend", "").title())
        #         col2.metric("Monthly Δ",  f"{model_info.get('slope', 0):+,.0f}")
        #         col3.metric("Confidence", summary.get("confidence", "").title(),
        #                     delta=f"R²={model_info.get('r2_score', 0)}")

        #         # ── Build rows — use sequential order, unique labels ──────────────
        #         rows = []

        #         for row in actuals:
        #             rows.append({
        #                 "order":  row["month"],                          # 1-12
        #                 "label":  f"{row['month_name']} {train_year}",  # "Jan 2023"
        #                 "value":  row["actual"],
        #                 "series": f"{train_year} Actual",
        #             })

        #         for i, row in enumerate(forecast):
        #             rows.append({
        #                 "order":  12 + (i + 1),                                      # 13, 14, 15
        #                 "label":  f"{row['month_name']} {forecast_year}",            # "Jan 2024"
        #                 "value":  row["forecasted_value"],
        #                 "series": f"{forecast_year} Forecast",
        #             })

        #         plot_df = pd.DataFrame(rows).sort_values("order")

        #         # ── Plotly — order locked via categoryarray ───────────────────────
        #         st.markdown(f"**{train_year} Actuals vs {forecast_year} Forecast**")

        #         fig = px.line(
        #             plot_df,
        #             x="label",
        #             y="value",
        #             color="series",
        #             markers=True,
        #             labels={
        #                 "label":  "Month",
        #                 "value":  metric.title(),
        #                 "series": "",
        #             },
        #             color_discrete_map={
        #                 f"{train_year} Actual":      "#4A90D9",
        #                 f"{forecast_year} Forecast": "#FF6B6B",
        #             },
        #         )

        #         fig.update_xaxes(
        #             categoryorder="array",
        #             categoryarray=plot_df["label"].tolist(),  # ← 15 unique labels in order
        #         )

        #         st.plotly_chart(fig, width="content")

        #         # ── Forecast table ────────────────────────────────────────────────
        #         st.markdown(f"**{forecast_year} Monthly Forecast**")
        #         st.dataframe(
        #             pd.DataFrame([
        #                 {
        #                     "Month":                f"{f['month_name']} {forecast_year}",
        #                     f"Forecast ({metric})": f"{f['forecasted_value']:,.0f}",
        #                 }
        #                 for f in forecast
        #             ]),
        #             width="content",
        #             hide_index=True,
        #         )
                
        # def on_tool_invoked_render_part(tool_name: str, tool_args: dict, tool_result: Any):
        #     """Tool-name based renderer."""

        #     if tool_name == "product_query":
        #         with result_container:
        #             st.markdown(":blue[🗄️ **Generated SQL:**]")
        #             st.code(
        #                 tool_args["sql"],
        #                 language="sql",
        #                 wrap_lines=True,
        #             )

        #     elif tool_name == "render_chart":
        #         on_tool_invoked_render_chart(tool_args, tool_result)

        #     elif tool_name == "render_sales_kpi":
        #         on_tool_invoked_render_kpi(tool_args, tool_result)
        #     elif tool_name == "sales_anomaly_detector":
        #         on_tool_invoked_render_anomaly(tool_args, tool_result)
        #     elif tool_name == "sales_forecast":                         # ← add
        #         on_tool_invoked_render_forecast(tool_args, tool_result)

        ##
        # def on_tool_invoked(tool_name: str, tool_args: dict, tool_result: Any):
        #     """Fired after each tool execution."""
        #     accumulated["text"] += (
        #         f"\n\n:blue[🔧 **Tool:** `{tool_name}`]\n"
        #         f"```json\n{json.dumps(tool_args, indent=2)}\n```\n"
        #         f"**Result:** `{tool_result}`\n\n"
        #     )

        #     result_area.markdown(accumulated["text"])

        #     on_tool_invoked_render_part(tool_name, tool_args, tool_result)

        # ── on_tool_invoked — now 3 lines ─────────────────────────────────────────────
        def on_tool_invoked(tool_name: str, tool_args: dict, tool_result: Any):
            accumulated["text"] += (
                f"\n\n:blue[🔧 **Tool:** `{tool_name}`]\n"
                f"```json\n{json.dumps(tool_args, indent=2)}\n```\n"
                f"**Result:** `{tool_result}`\n\n"
            )
            result_area.markdown(accumulated["text"])
            renderer_registry.render(tool_name, tool_args, tool_result, result_container)

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

        with st.spinner("Processing...", show_time=True, width="content"):
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