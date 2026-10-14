# Converse Tool Architecture Documentation

## Overview

This document provides comprehensive documentation of the Converse Tool implementation in the streamlit-demo project, focusing on the architectural evolution from version 3.7.4 to 3.7.5.

## File Locations

### Main Application Files
- **v3.7.4**: `pages/3_7_4_converse_tool_demo.py` - Prototype implementation
- **v3.7.5**: `pages/3_7_5_converse_tool_v3.py` - Production-ready refactored version

### Core Library Modules
```
cmn/
├── bedrock/
│   ├── client_manager.py          # BedrockClientFactory for boto3 client creation
│   └── converse/
│       ├── __init__.py            # Exports ConversationManager, StreamResult, ...
│       ├── conversation_history.py # Trim / validate / strip-attachments for message histories
│       ├── conversation_manager.py # Orchestrates multi-turn conversations
│       ├── models.py              # StreamResult, ToolInvocation, StreamMetrics dataclasses
│       ├── request_builder.py     # converse_stream() kwargs from a FoundationModel + InferenceSettings
│       └── stream_processor.py    # Parses AWS Bedrock event streams
├── bedrock_models.py              # FOUNDATION_MODELS catalog (see design/model-catalog.md)
├── tools/
│   ├── tool/
│   │   ├── __init__.py            # ToolRegistry + all tool exports
│   │   ├── bedrock_converse_tools_tool.py          # Base BedrockConverseTool class
│   │   ├── bedrock_converse_tools_tool_calculator.py
│   │   ├── bedrock_converse_tools_tool_acronym.py
│   │   ├── bedrock_converse_tools_tool_url.py
│   │   ├── bedrock_converse_tools_tool_wikipedia.py
│   │   ├── bedrock_converse_tools_tool_datetime.py
│   │   ├── bedrock_converse_tools_tool_sales.py
│   │   ├── bedrock_converse_tools_tool_product.py
│   │   ├── bedrock_converse_tools_tool_chart.py
│   │   ├── bedrock_converse_tools_tool_sales_kpi.py
│   │   ├── bedrock_converse_tools_tool_sales_forecast.py
│   │   ├── bedrock_converse_tools_tool_sales_anomaly.py
│   │   ├── bedrock_converse_tools_tool_pdf.py
│   │   ├── bedrock_converse_tools_tool_pptx.py
│   │   └── ...
│   └── renderer/
│       ├── __init__.py            # RendererRegistry + all renderer exports
│       ├── bedrock_converse_tools_renderer.py      # Base renderer class
│       ├── bedrock_converse_tools_renderer_chart.py
│       ├── bedrock_converse_tools_renderer_product.py
│       ├── bedrock_converse_tools_renderer_sales_kpi.py
│       ├── bedrock_converse_tools_renderer_sales_anomaly.py
│       ├── bedrock_converse_tools_renderer_sales_forecast.py
│       ├── bedrock_converse_tools_renderer_pdf.py
│       ├── bedrock_converse_tools_renderer_pptx.py
│       └── ...
└── view/
    ├── mime_constants.py          # MIME type mappings
    ├── __init__.py               # CONVERSE_TOOL_GUIDE export
    └── processor/
        └── file_uploader_chat.py  # render_file_uploader() function
```

---

## Architecture Comparison: v3.7.4 vs v3.7.5

### 1. Tool Handling - Single vs Multi-Tool

#### v3.7.4 (Prototype)
- **Limitation**: Handles **only ONE tool call** per turn
- **Flow**:
  1. User sends message
  2. LLM responds with text OR one tool call
  3. If tool call detected:
     - Execute the ONE tool
     - Build tool result message
     - Call `invoke_llm()` ONCE more
     - Done - no further loops
  4. Save assistant message to session state

**Code Snippet (v3.7.4)**:
```python
if tool_invocation['tool_name'] != None:
    tool_name = tool_invocation['tool_name']
    tool_args_json = json.loads(tool_invocation['tool_arguments'])
    
    # Manual iteration to find matching tool
    for tool in tools:
        if tool.matches(tool_name):
            expr_result = tool.invoke(tool_args_json['expression'], tool_args=tool_args_json)
            tool_result_message = {
                "role": "user",
                "content": [{
                    "toolResult": {
                        "toolUseId": tool_invocation['tool_use_id'],
                        "content": [{"json": {"expr_result": expr_result}}]
                    }
                }]
            }
            break
    
    messages = [message_user_latest, tool_request_message, tool_result_message]
    invoke_llm(messages, system_prompts, tool_config, inference_config, additional_model_fields)
```

**Limitations**:
- Cannot handle multiple tool calls in sequence
- Cannot handle iterative tool loops (tool → LLM → tool → LLM → ...)
- Manual tool matching logic
- Hardcoded result format (`expr_result`)

#### v3.7.5 (Production)
- **Capability**: Handles **MULTIPLE tool calls** automatically in a loop
- **Flow**:
  1. User sends message
  2. **While loop begins**:
     - Call LLM
     - If response contains tool calls:
       - Execute ALL tools in parallel concept (sequentially in code)
       - Collect ALL tool results
       - Append assistant message (with all tool uses)
       - Append user message (with all tool results)
       - **Loop continues** - LLM sees all results
     - If no tool calls, exit loop
  3. Save final assistant message

**Code Snippet (v3.7.5 - ConversationManager)**:
```python
def run(self, message_history, on_text_delta, on_stream_result, on_tool_invoked) -> StreamResult:
    messages = message_history.copy()
    
    while True:  # ← Automatic tool loop
        result = self._call_llm(messages, on_text_delta)
        
        if on_stream_result:
            on_stream_result(result)
        
        if not result.has_tool_call:
            return result  # Exit when no more tools needed
        
        # ── Build assistant message for ALL tool calls ────────
        assistant_content = [
            {
                "toolUse": {
                    "toolUseId": tool_inv.tool_use_id,
                    "name":      tool_inv.tool_name,
                    "input":     tool_inv.tool_arguments,
                }
            }
            for tool_inv in result.tool_invocations  # ← ALL tools
        ]
        messages.append({"role": "assistant", "content": assistant_content})
        
        # ── Execute each tool + collect results ───────────────
        tool_results_content = []
        
        for tool_inv in result.tool_invocations:
            tool_result = self.registry.invoke(  # ← Registry pattern
                tool_inv.tool_name,
                tool_inv.tool_arguments,
            )
            
            if on_tool_invoked:
                on_tool_invoked(tool_inv.tool_name, tool_inv.tool_arguments, tool_result)
            
            tool_results_content.append({
                "toolResult": {
                    "toolUseId": tool_inv.tool_use_id,
                    "content":   [{"json": {"result": tool_result}}],  # ← Generic format
                }
            })
        
        # ── Single user message with all tool results ─────────
        messages.append({"role": "user", "content": tool_results_content})
        # Loop continues → LLM sees all results
```

**Advantages**:
- Supports complex multi-step tool chains
- Supports multiple tools per turn
- Generic tool result format
- Clean separation via ConversationManager

---

### 2. Separation of Concerns

#### v3.7.4 Architecture
**Monolithic Design** - Everything in one file:
- Stream parsing logic (contentBlockStart, contentBlockDelta, messageStop)
- Tool invocation logic
- UI rendering (st.chat_message, st.write, st.markdown)
- Error handling
- Session state management

**Problems**:
- Difficult to test (tightly coupled to Streamlit)
- Cannot reuse logic in other contexts (CLI, API, batch jobs)
- Hard to maintain and extend
- Business logic mixed with presentation

#### v3.7.5 Architecture
**Layered Design** - Clean separation:

```
┌─────────────────────────────────────────────────────┐
│  Presentation Layer (3_7_5_converse_tool_v3.py)    │
│  - Streamlit UI                                     │
│  - Session state management                         │
│  - Callbacks: on_text_delta, on_stream_result,     │
│    on_tool_invoked                                  │
│  - Uses: ConversationManager, ToolRegistry,         │
│    RendererRegistry                                 │
└─────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│  Business Logic Layer                               │
│  ┌───────────────────────────────────────────────┐ │
│  │ ConversationManager (no Streamlit imports)    │ │
│  │ - Orchestrates multi-turn conversations       │ │
│  │ - Manages tool loops                          │ │
│  │ - Calls: StreamProcessor, ToolRegistry        │ │
│  └───────────────────────────────────────────────┘ │
│  ┌───────────────────────────────────────────────┐ │
│  │ StreamProcessor (no Streamlit imports)        │ │
│  │ - Parses AWS Bedrock event streams            │ │
│  │ - Extracts text, tool calls, metrics          │ │
│  │ - Returns: StreamResult with all data         │ │
│  └───────────────────────────────────────────────┘ │
│  ┌───────────────────────────────────────────────┐ │
│  │ ToolRegistry (no Streamlit imports)           │ │
│  │ - Manages tool definitions                    │ │
│  │ - Invokes tools by name                       │ │
│  │ - Generates tool_config for Bedrock           │ │
│  └───────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│  Rendering Layer (Streamlit-aware)                  │
│  ┌───────────────────────────────────────────────┐ │
│  │ RendererRegistry                              │ │
│  │ - Maps tool names to renderers                │ │
│  │ - Renders charts, tables, rich UI            │ │
│  └───────────────────────────────────────────────┘ │
│  ┌───────────────────────────────────────────────┐ │
│  │ Specific Renderers (ChartToolRenderer, etc.) │ │
│  │ - Calls st.plotly_chart(), st.dataframe()     │ │
│  └───────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────┘
```

**Benefits**:
- **Testable**: Business logic can be unit tested without Streamlit
- **Reusable**: ConversationManager can be used in CLI, API, Lambda
- **Maintainable**: Each component has single responsibility
- **Extensible**: Easy to add new tools/renderers without touching core logic

---

### 3. Invocation Metrics Tracking

#### v3.7.4 Metrics
**Simple per-call tracking**:
```python
if settings["enable_print_invocation_metrics"]:
    stats = f"T| token.in={input_token_count} token.out={output_token_count} token={total_token_count} latency={latency}"
    result_container.write(stats)
```

**Limitations**:
- Shows only last LLM call metrics
- Doesn't accumulate across tool loops
- No history of tools called
- Lost after page refresh

#### v3.7.5 Metrics
**Cumulative turn-level tracking with InvocationStat**:

```python
@dataclass
class InvocationStat:
    input_tokens:  int = 0
    output_tokens: int = 0
    total_tokens:  int = 0
    latency_ms:    int = 0
    llm_calls:     int = 0           # ← Count of LLM invocations
    tools_called:  list = field(default_factory=list)  # ← Tool sequence
    
    def accumulate(self, result: StreamResult) -> None:
        """Accumulates metrics from multiple LLM calls in one turn."""
        m = result.metrics
        self.input_tokens  += m.input_tokens
        self.output_tokens += m.output_tokens
        self.total_tokens  += m.total_tokens
        self.latency_ms    += m.latency_ms
        self.llm_calls     += 1
    
    def record_tool(self, tool_name: str) -> None:
        """Records each tool invocation."""
        self.tools_called.append(tool_name)
    
    def as_markdown(self) -> str:
        """Formats metrics for display."""
        lines = [
            f"🔢 in={self.input_tokens} out={self.output_tokens} "
            f"total={self.total_tokens} latency={self.latency_ms}ms "
            f"calls={self.llm_calls}"
        ]
        if self.tools_called:
            lines.append(f"🔧 tools: {', '.join(f'`{n}`' for n in self.tools_called)}")
        return "  \n".join(lines)
```

**Usage in v3.7.5**:
```python
# Per-turn accumulator
turn_stat = InvocationStat()

def on_stream_result(result: StreamResult):
    turn_stat.accumulate(result)  # ← Accumulates across all LLM calls

def on_tool_invoked(tool_name: str, tool_args: dict, tool_result: Any):
    turn_stat.record_tool(tool_name)  # ← Records tool sequence

# After conversation completes
st.session_state.invocation_stats.append(turn_stat)  # ← Persisted
```

**Display on message re-render**:
```python
for idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(text)
        
        if opt_show_metrics and msg["role"] == "assistant":
            stat = st.session_state.invocation_stats[idx]
            if stat is not None:
                st.caption(stat.as_markdown())  # ← Shows cumulative metrics
```

**Advantages**:
- **Complete picture**: Shows total cost of entire turn (all LLM calls + tools)
- **Tool audit trail**: See which tools were called in what order
- **Persistent**: Metrics survive page refresh
- **Per-message**: Each assistant message has its own stat object

**Example Output**:
```
🔢 in=1523 out=487 total=2010 latency=3421ms calls=3
🔧 tools: `sales_data`, `calculator`, `chart`
```
This shows the turn made 3 LLM calls (initial + 2 tool loops) and used 3 tools.

---

### 4. Tool Registry Pattern

#### v3.7.4 - Manual Tool Management
**Hardcoded list**:
```python
calculator_tool = CalculatorBedrockConverseTool()
acronym_tool = AcronymBedrockConverseTool()
url_loader_tool = UrlContentBedrockConverseTool()
wikipedia_tool = WikipediaBedrockConverseTool()
datetime_tool = DateTimeBedrockConverseTool()

tools = [calculator_tool, acronym_tool, url_loader_tool, wikipedia_tool, datetime_tool]

tool_config = {
    "tools": [tool.definition for tool in tools]
}
```

**Manual tool invocation**:
```python
for tool in tools:
    if tool.matches(tool_name):
        expr_result = tool.invoke(tool_args_json['expression'], tool_args=tool_args_json)
        # ... build result message
        break
```

**Problems**:
- Must manually maintain tool list
- Must manually iterate to find tool
- Tool-specific invoke logic (passing `'expression'` key)
- Difficult to enable/disable tools dynamically

#### v3.7.5 - ToolRegistry Pattern
**Centralized registry**:
```python
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
        PptxBedrockConverseTool(),
        PdfBedrockConverseTool(),
    ])

tool_registry = get_tool_registry()
```

**ToolRegistry class capabilities**:
```python
class ToolRegistry:
    def __init__(self, tools: list):
        self._tools = {tool.name: tool for tool in tools}
    
    @property
    def tool_config(self) -> dict:
        """Generates Bedrock tool_config automatically."""
        return {
            "tools": [tool.definition for tool in self._tools.values()]
        }
    
    @property
    def tool_names(self) -> list:
        """Returns list of tool names for display."""
        return list(self._tools.keys())
    
    def invoke(self, tool_name: str, tool_arguments: dict) -> Any:
        """Generic tool invocation - no tool-specific logic needed."""
        if tool_name not in self._tools:
            raise ValueError(f"Unknown tool: {tool_name}")
        return self._tools[tool_name].invoke(**tool_arguments)
    
    def build_tool_summary(self) -> str:
        """Generates tool documentation for system prompt."""
        summaries = []
        for tool in self._tools.values():
            summaries.append(f"- {tool.name}: {tool.description}")
        return "Available tools:\n" + "\n".join(summaries)
```

**Usage**:
```python
# Automatic tool config generation
manager = ConversationManager(
    tool_registry=tool_registry,
    # ...
)

# Generic invocation inside ConversationManager
tool_result = self.registry.invoke(tool_name, tool_arguments)
```

**Dynamic system prompt**:
```python
def build_default_system_prompt(registry: ToolRegistry) -> str:
    tool_summary = registry.build_tool_summary()
    return "\n\n".join(filter(None, [
        "You are a BI analyst assistant.",
        tool_summary,  # ← Automatically includes all tool descriptions
        "Call tools ONE AT A TIME. Wait for each result before calling the next.",
    ]))
```

**Advantages**:
- Single source of truth for tools
- Easy to add/remove tools (just modify list)
- No tool-specific invocation logic
- Auto-generates tool config and documentation
- Cached via `@st.cache_resource` for performance

---

### 5. Stream Processing

#### v3.7.4 - Inline Stream Parsing
**Direct event handling in UI code**:
```python
stream = response.get('stream')
for event in stream:
    if 'messageStart' in event:
        pass
    
    if 'contentBlockStart' in event:
        content_block_start = event['contentBlockStart']
        if 'start' in content_block_start:
            content_block_start_start = content_block_start['start']
            if 'toolUse' in content_block_start_start:
                content_block_tool_use = content_block_start_start['toolUse']
                tool_use_id = content_block_tool_use['toolUseId']
                tool_use_name = content_block_tool_use['name']
                tool_invocation['tool_name'] = tool_use_name
                tool_invocation['tool_use_id'] = tool_use_id
    
    if 'contentBlockDelta' in event:
        content_delta = event['contentBlockDelta']['delta']
        if 'text' in content_delta:
            result_text += f"{content_delta['text']}"
            result_area.write(result_text)  # ← UI logic mixed in
        if 'toolUse' in content_delta:
            content_delta_tool_input = content_delta['toolUse']['input']
            tool_input += content_delta_tool_input
    
    if 'messageStop' in event:
        stop_reason = event['messageStop']['stopReason']
        if stop_reason == 'tool_use':
            tool_input_json = json.loads(tool_input)
            tool_invocation['tool_arguments'] = tool_input
    
    # ... more event handling
```

**Problems**:
- Stream parsing logic mixed with UI rendering
- Cannot test parsing logic independently
- Cannot reuse in non-UI contexts
- Hard to maintain (deeply nested conditions)

#### v3.7.5 - StreamProcessor Module
**Separated stream parsing** (`cmn/bedrock/converse/stream_processor.py`):

```python
def process_stream(
    stream,
    on_text_delta: Optional[Callable[[str], None]] = None,
) -> StreamResult:
    """
    Pure EventStream parser — no UI dependencies.
    
    Returns
    -------
    StreamResult containing:
        - text: accumulated text response
        - stop_reason: end_turn, tool_use, max_tokens, etc.
        - tool_invocations: list of ToolInvocation objects
        - metrics: StreamMetrics with token counts and latency
        - errors: list of error messages
    """
    result           = StreamResult()
    tool_invocations = []
    current_tool     = None
    
    for event in stream:
        if 'messageStart' in event:
            logger.debug("messageStart: role=%s", event['messageStart'].get('role'))
        
        elif 'contentBlockStart' in event:
            start = event['contentBlockStart'].get('start', {})
            if 'toolUse' in start:
                # ── New tool call — create fresh ToolInvocation ───
                tu = start['toolUse']
                current_tool = ToolInvocation(
                    tool_use_id = tu['toolUseId'],
                    tool_name   = tu['name'],
                )
                tool_invocations.append(current_tool)
        
        elif 'contentBlockDelta' in event:
            delta = event['contentBlockDelta']['delta']
            
            if 'text' in delta:
                chunk = delta['text']
                result.text += chunk
                if on_text_delta:
                    on_text_delta(chunk)  # ← Callback to UI layer
            
            elif 'toolUse' in delta:
                # ── Accumulate into CURRENT tool only ─────────
                if current_tool is not None:
                    current_tool.tool_input_raw += delta['toolUse'].get('input', '')
        
        elif 'contentBlockStop' in event:
            # ── Block done — finalize current tool ────────
            if current_tool is not None:
                current_tool.finalize()  # Parses JSON
                current_tool = None
        
        elif 'messageStop' in event:
            result.stop_reason = event['messageStop'].get('stopReason', '')
            if result.stop_reason == 'tool_use' and tool_invocations:
                result.tool_invocations = tool_invocations
        
        elif 'metadata' in event:
            _handle_metadata(event['metadata'], result)
        
        else:
            # Handle exceptions (internalServerException, etc.)
            for exc_key in _EXCEPTION_EVENTS:
                if exc_key in event:
                    msg = event[exc_key].get('message', exc_key)
                    result.errors.append(f"[{exc_key}] {msg}")
    
    return result
```

**Data Models** (`cmn/bedrock/converse/models.py`):
```python
@dataclass
class ToolInvocation:
    """Captures a single tool call requested by the model."""
    tool_use_id:    str = None
    tool_name:      str = None
    tool_input_raw: str = ""      # accumulated JSON string from stream deltas
    tool_arguments: dict = None   # parsed after stream ends
    
    def finalize(self):
        """Parse raw input string into dict. Call once after stream ends."""
        if self.tool_input_raw:
            self.tool_arguments = json.loads(self.tool_input_raw)
        else:
            self.tool_arguments = {}
        return self

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
    tool_invocations: list = field(default_factory=list)
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
```

**Advantages**:
- **Pure function**: Takes stream, returns structured data
- **Testable**: Can unit test with mock streams
- **Reusable**: Same parser works in UI, CLI, Lambda
- **Clean data**: All parsing logic encapsulated in one place
- **Type-safe**: Uses dataclasses for structured results

---

### 6. Renderer Registry Pattern

#### v3.7.4 - No Rich Rendering
**Simple text display**:
```python
result_text += f"\n\n:blue[Tool Request: {json.dumps(tool_request_message, indent=2)}]\n"
result_area.markdown(result_text)

result_text += f"\n\nTool Result: {json.dumps(tool_result_message, indent=2)}\n"
result_area.markdown(result_text)
```

**Limitations**:
- Only shows JSON dumps
- No charts, tables, or rich visualizations
- User must parse JSON manually

#### v3.7.5 - RendererRegistry + Custom Renderers
**Renderer Registry**:
```python
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
```

**RendererRegistry class**:
```python
class RendererRegistry:
    def __init__(self, renderers: list):
        self._renderers = {r.tool_name: r for r in renderers}
    
    def render(self, tool_name: str, tool_args: dict, tool_result: Any, container):
        """
        Render tool result using custom renderer if available.
        
        Parameters
        ----------
        tool_name : str
            Name of the tool that was invoked
        tool_args : dict
            Arguments passed to the tool
        tool_result : Any
            Result returned by the tool
        container : streamlit container
            Where to render the output
        """
        if tool_name in self._renderers:
            self._renderers[tool_name].render(tool_args, tool_result, container)
```

**Example Renderer - ChartToolRenderer**:
```python
class ChartToolRenderer(BedrockConverseToolRenderer):
    def __init__(self):
        super().__init__(tool_name="chart")
    
    def render(self, tool_args: dict, tool_result: Any, container):
        """Renders Plotly charts in Streamlit."""
        if isinstance(tool_result, dict) and 'figure' in tool_result:
            fig = tool_result['figure']
            container.plotly_chart(fig, use_container_width=True)
        elif isinstance(tool_result, dict) and 'dataframe' in tool_result:
            df = pd.DataFrame(tool_result['dataframe'])
            container.dataframe(df)
```

**Usage in conversation callbacks**:
```python
def on_tool_invoked(tool_name: str, tool_args: dict, tool_result: Any):
    # Show tool annotation in text
    accumulated["text"] += (
        f"\n\n:blue[🔧 **Tool:** `{tool_name}`]\n"
        f"```json\n{json.dumps(tool_args, indent=2)}\n```\n"
        f"**Result:** `{tool_result}`\n\n"
    )
    result_area.markdown(accumulated["text"])
    
    # Render rich visualization if renderer available
    renderer_registry.render(tool_name, tool_args, tool_result, result_container)
```

**Advantages**:
- **Rich visualizations**: Charts, tables, images, documents
- **Extensible**: Easy to add new renderers
- **Decoupled**: Renderer logic separate from tool logic
- **Optional**: Falls back to JSON if no renderer available

**Supported Renderers**:
- **ChartToolRenderer**: Plotly charts (line, bar, scatter, etc.)
- **ProductToolRenderer**: Product cards with images
- **SalesKpiToolRenderer**: KPI metrics with formatting
- **SalesAnomalyToolRenderer**: Anomaly detection visualizations
- **SalesForecastToolRenderer**: Forecast charts with confidence intervals
- **PptxToolRenderer**: PowerPoint file download links
- **PdfToolRenderer**: PDF file previews and downloads

---

### 7. Error Handling

#### v3.7.4 - Inline Error Handling
```python
if stop_reason not in ('end_turn', 'tool_use'):
    stop_reason_display = stop_reason
    if stop_reason == 'max_tokens':
        stop_reason_display = "Insufficient Tokens. Increaes MaxToken Settings."
    result_text_error = f"{result_text}\n\n:red[Generation Stopped: {stop_reason_display}]"
    result_area.write(result_text_error)

if "internalServerException" in event:
    exception = event["internalServerException"]
    result_text += f"\n\{exception}"
    result_area.write(result_text)
# ... repeated for each exception type
```

**Problems**:
- Error detection mixed with display logic
- No centralized error collection
- Cannot inspect errors after stream completes

#### v3.7.5 - Structured Error Handling
**Error collection in StreamProcessor**:
```python
_EXCEPTION_EVENTS = [
    "internalServerException",
    "modelStreamErrorException",
    "throttlingException",
    "validationException",
]

# Inside process_stream()
for exc_key in _EXCEPTION_EVENTS:
    if exc_key in event:
        msg = event[exc_key].get('message', exc_key)
        logger.error("Stream exception [%s]: %s", exc_key, msg)
        result.errors.append(f"[{exc_key}] {msg}")
```

**Error display in UI callback**:
```python
def on_stream_result(result: StreamResult):
    turn_stat.accumulate(result)
    
    if result.stop_reason_display:
        result_area.markdown(
            f"{accumulated['text']}\n\n"
            f":red[Generation Stopped: {result.stop_reason_display}]"
        )
    
    for err in result.errors:
        st.error(err)  # ← Clean error display
```

**Stop reason mapping**:
```python
STOP_REASON_MESSAGES = {
    "max_tokens":           "Insufficient Tokens. Increase MaxToken Settings.",
    "guardrail_intervened": "Response blocked by guardrail.",
    "content_filtered":     "Content was filtered.",
}

@property
def stop_reason_display(self) -> Optional[str]:
    """Human-readable stop reason, or None if normal end_turn / tool_use."""
    if self.stop_reason in ("end_turn", "tool_use"):
        return None
    return STOP_REASON_MESSAGES.get(self.stop_reason, self.stop_reason)
```

**Advantages**:
- Centralized error collection
- Errors available after stream completes
- Clean separation: detection vs display
- Consistent error formatting

---

### 8. File Upload Handling

#### v3.7.4 - Inline Upload Processing
```python
uploaded_file = st.file_uploader(
    "Attach Image",
    type=["PNG", "JPEG", "TXT", "CSV", "PDF", "MD"],
    accept_multiple_files=False,
    label_visibility="collapsed",
    key=f"menu_converse_tool_uploader_key_{st.session_state.menu_converse_tool_uploader_key}"
)

# Inline processing logic
if uploaded_file:
    if uploaded_file.type in mime_mapping_image:
        uploaded_file_bytes = uploaded_file.read()
        image:Image = Image.open(uploaded_file)
        uploaded_file_name = uploaded_file.name
        uploaded_file_type = uploaded_file.type
        uploaded_file_base64 = image_to_base64(image, mime_mapping[uploaded_file_type])
        st.image(image, caption='upload images', use_column_width=True)
    elif uploaded_file.type in mime_mapping_document:
        # ... more inline processing
```

#### v3.7.5 - Modular Upload Component
**Extracted component** (`cmn/view/processor/file_uploader_chat.py`):
```python
uploaded_file, file_bytes, file_key, file_type, file_preview = render_file_uploader(
    st.session_state.uploader_key
)
```

**Benefits**:
- Reusable across pages
- Cleaner main file
- Easier to maintain file handling logic

---

### 9. System Prompt

#### v3.7.4 - Static System Prompt
```python
opt_system_msg = st.text_area(
    label="System Message", 
    value="You are a question and answering chatbot", 
    key="system_msg"
)
```

**Limitations**:
- No tool documentation in system prompt
- User must manually describe tools
- Tools and prompt can get out of sync

#### v3.7.5 - Dynamic System Prompt
```python
def build_default_system_prompt(registry: ToolRegistry) -> str:
    tool_summary = registry.build_tool_summary()
    return "\n\n".join(filter(None, [
        "You are a BI analyst assistant.",
        tool_summary,  # ← Auto-generated from registry
        "Call tools ONE AT A TIME. Wait for each result before calling the next.",
    ]))

OPT_SYSTEM_MSG_DEFAULT = build_default_system_prompt(tool_registry)
```

**ToolRegistry.build_tool_summary()**:
```python
def build_tool_summary(self) -> str:
    """Generates tool documentation for system prompt."""
    summaries = []
    for tool in self._tools.values():
        summaries.append(f"- {tool.name}: {tool.description}")
    return "Available tools:\n" + "\n".join(summaries)
```

**Example Generated Prompt**:
```
You are a BI analyst assistant.

Available tools:
- calculator: Performs mathematical calculations
- acronym: Expands acronyms and abbreviations
- url_content: Fetches and extracts content from URLs
- wikipedia: Searches Wikipedia for information
- aws_docs: Searches AWS documentation
- datetime: Provides current date and time
- sales_data: Queries sales database
- product: Searches product catalog
- chart: Creates charts and visualizations
- sales_kpi: Calculates sales KPIs
- sales_forecast: Forecasts sales trends
- sales_anomaly: Detects sales anomalies
- pptx: Generates PowerPoint presentations
- pdf: Generates PDF documents

Call tools ONE AT A TIME. Wait for each result before calling the next.
```

**Advantages**:
- Always up-to-date with available tools
- Consistent tool descriptions
- Add tool → automatically in system prompt
- No manual prompt maintenance

---

## Key Data Structures

### StreamResult (v3.7.5)
```python
@dataclass
class StreamResult:
    text:             str = ""
    stop_reason:      str = ""
    tool_invocations: list = field(default_factory=list)  # List[ToolInvocation]
    metrics:          StreamMetrics = field(default_factory=StreamMetrics)
    errors:           list = field(default_factory=list)
    
    @property
    def has_tool_call(self) -> bool:
        return self.stop_reason == "tool_use" and bool(self.tool_invocations)
```

### ToolInvocation (v3.7.5)
```python
@dataclass
class ToolInvocation:
    tool_use_id:    str = None
    tool_name:      str = None
    tool_input_raw: str = ""      # JSON string accumulated from stream
    tool_arguments: dict = None   # Parsed JSON
    
    def finalize(self):
        if self.tool_input_raw:
            self.tool_arguments = json.loads(self.tool_input_raw)
        else:
            self.tool_arguments = {}
        return self
```

### InvocationStat (v3.7.5)
```python
@dataclass
class InvocationStat:
    input_tokens:  int = 0
    output_tokens: int = 0
    total_tokens:  int = 0
    latency_ms:    int = 0
    llm_calls:     int = 0
    tools_called:  list = field(default_factory=list)
    
    def accumulate(self, result: StreamResult) -> None:
        """Accumulate metrics from one LLM call."""
        pass
    
    def record_tool(self, tool_name: str) -> None:
        """Record tool invocation."""
        pass
    
    def as_markdown(self) -> str:
        """Format for display."""
        pass
```

---

## Session State Structure (v3.7.5)

```python
st.session_state.messages = [
    {
        "role": "user",
        "content": [
            {"text": "What is 5 + 3?"}
        ]
    },
    {
        "role": "assistant",
        "content": [
            {"text": "Let me calculate that...\n\nResult: 8"}
        ]
    },
    # ... more messages
]

st.session_state.invocation_stats = [
    None,  # No stats for user messages
    InvocationStat(input_tokens=123, output_tokens=45, ..., tools_called=['calculator']),
    None,
    InvocationStat(input_tokens=234, output_tokens=67, ..., tools_called=[]),
    # ... parallel array with messages
]

st.session_state.uploader_key = 0  # Increments to reset file uploader
```

---

## Message Format

### User Message with Document
```python
{
    "role": "user",
    "content": [
        {"text": "Analyze this CSV"},
        {
            "document": {
                "format": "csv",
                "name": "sales_data_2024_csv",
                "source": {
                    "bytes": b"..."  # base64-encoded bytes
                }
            }
        }
    ]
}
```

### Assistant Message with Tool Use
```python
{
    "role": "assistant",
    "content": [
        {
            "toolUse": {
                "toolUseId": "tooluse_abc123",
                "name": "sales_data",
                "input": {"year": 2024, "region": "US"}
            }
        }
    ]
}
```

### User Message with Tool Result
```python
{
    "role": "user",
    "content": [
        {
            "toolResult": {
                "toolUseId": "tooluse_abc123",
                "content": [
                    {"json": {"result": {"revenue": 1000000, "units": 5000}}}
                ]
            }
        }
    ]
}
```

---

## Callback Pattern (v3.7.5)

The v3.7.5 architecture uses callbacks to decouple business logic from UI:

```python
# Define UI callbacks
def on_text_delta(chunk: str):
    """Fired for every streamed text chunk."""
    accumulated["text"] += chunk
    result_area.markdown(accumulated["text"])

def on_stream_result(result: StreamResult):
    """Fired after each complete LLM response."""
    turn_stat.accumulate(result)
    if result.stop_reason_display:
        result_area.markdown(f"{accumulated['text']}\n\n:red[{result.stop_reason_display}]")
    for err in result.errors:
        st.error(err)

def on_tool_invoked(tool_name: str, tool_args: dict, tool_result: Any):
    """Fired after each tool execution."""
    turn_stat.record_tool(tool_name)
    accumulated["text"] += f"\n\n:blue[🔧 **Tool:** `{tool_name}`]\n..."
    result_area.markdown(accumulated["text"])
    renderer_registry.render(tool_name, tool_args, tool_result, result_container)

# Pass callbacks to ConversationManager
manager = ConversationManager(...)
final_result = manager.run(
    message_history,
    on_text_delta=on_text_delta,
    on_stream_result=on_stream_result,
    on_tool_invoked=on_tool_invoked,
)
```

**Why Callbacks?**
- ConversationManager has no Streamlit imports
- UI layer controls how to display results
- Easy to swap UI (CLI, web, desktop)
- Testable (can pass mock callbacks)

---

## Tool Loop Flow (v3.7.5)

```
User Message
    ↓
┌─────────────────────────────────────────┐
│ ConversationManager.run()              │
│                                         │
│ while True:                             │
│   ┌───────────────────────────────────┐ │
│   │ 1. Call LLM (converse_stream)     │ │
│   │    - StreamProcessor parses stream│ │
│   │    - Returns StreamResult          │ │
│   └───────────────────────────────────┘ │
│            ↓                            │
│   ┌───────────────────────────────────┐ │
│   │ 2. Callback: on_stream_result()   │ │
│   │    - Update UI with text          │ │
│   │    - Accumulate metrics           │ │
│   │    - Show errors                  │ │
│   └───────────────────────────────────┘ │
│            ↓                            │
│   ┌───────────────────────────────────┐ │
│   │ 3. Check: has_tool_call?          │ │
│   └───────────────────────────────────┘ │
│      ↓ No                      ↓ Yes    │
│   [EXIT]                    [CONTINUE]  │
│                                 ↓        │
│   ┌───────────────────────────────────┐ │
│   │ 4. Build assistant message        │ │
│   │    with ALL tool uses             │ │
│   └───────────────────────────────────┘ │
│            ↓                            │
│   ┌───────────────────────────────────┐ │
│   │ 5. Execute each tool              │ │
│   │    FOR each tool_invocation:      │ │
│   │      - registry.invoke()          │ │
│   │      - Callback: on_tool_invoked()│ │
│   │      - Collect result             │ │
│   └───────────────────────────────────┘ │
│            ↓                            │
│   ┌───────────────────────────────────┐ │
│   │ 6. Build user message             │ │
│   │    with ALL tool results          │ │
│   └───────────────────────────────────┘ │
│            ↓                            │
│   [LOOP BACK TO STEP 1]                 │
│                                         │
└─────────────────────────────────────────┘
```

**Example Scenario**: User asks "What is 5 + 3 multiplied by 2?"

1. **LLM Call 1**: "I'll calculate 5 + 3 first" → `tool_use: calculator(5+3)`
2. **Tool Execution**: `calculator.invoke(expression="5+3")` → `result: 8`
3. **LLM Call 2**: "Now I'll multiply by 2" → `tool_use: calculator(8*2)`
4. **Tool Execution**: `calculator.invoke(expression="8*2")` → `result: 16`
5. **LLM Call 3**: "The answer is 16" → `stop_reason: end_turn`
6. **Exit**: Return final result

**Total**: 3 LLM calls, 2 tool executions, all in one `manager.run()` invocation

---

## Configuration (v3.7.5)

### Model Configuration
```python
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
```

### Inference Configuration
```python
inference_config = {
    "temperature": opt_temperature,  # 0.0 - 1.0
    "maxTokens":   opt_max_tokens,   # 1 - 8192
}
```

### Additional Model Fields (Optional)
```python
additional_model_fields = {
    "top_p": 0.9,
    "top_k": 250,
}
```

### Building Requests from the Model Catalog

The examples above are the 3_7_x tool pages' own hand-built config. Newer
pages (`3_5_13_converse_demo.py`) build the request from the model catalog
instead, so a parameter the model rejects is never sent:

```python
from cmn.bedrock_models import FOUNDATION_MODELS
from cmn.bedrock.converse import InferenceSettings, build_converse_request, process_stream

model = FOUNDATION_MODELS.find("global.anthropic.claude-sonnet-4-6")
request = build_converse_request(
    model, message_history, system_prompt,
    InferenceSettings(max_tokens=4096, temperature=0.1, top_p=0.9, top_k=250),
)
# -> inferenceConfig {"maxTokens": 4096, "temperature": 0.1}
#    additionalModelRequestFields {"top_k": 250}   (Sonnet 4.6 rejects top_p alongside temperature)

response = bedrock_runtime.converse_stream(**request)
result = process_stream(response["stream"], on_text_delta=show_chunk)
if result.errors:
    ...  # show, but don't save the exchange to history
```

`additionalModelRequestFields` is omitted when empty. top_k is sent under the
model's native field name (`top_k`, `k`, `TopK`). See `design/model-catalog.md`
for the parameter sets.

---

## Performance Optimizations (v3.7.5)

### 1. Cached Resources
```python
@st.cache_resource
def get_bedrock_client():
    return BedrockClientFactory.bedrock_runtime(region=AWS_REGION)

@st.cache_resource
def get_tool_registry():
    return ToolRegistry([...])

@st.cache_resource
def get_renderer_registry():
    return RendererRegistry([...])
```

**Benefits**:
- Single boto3 client instance (reuses connections)
- Tools instantiated once
- Renderers instantiated once

### 2. Message History Trimming
```python
MAX_MESSAGES = 100 * 2  # 200 messages (100 turns)

# After each turn
msgs = st.session_state.messages
stats = st.session_state.invocation_stats
if len(msgs) > MAX_MESSAGES:
    excess = len(msgs) - MAX_MESSAGES
    del msgs[0:excess]
    del stats[0:excess]
```

**Benefits**:
- Prevents unbounded memory growth
- Maintains reasonable context window
- Keeps parallel arrays in sync

### 3. File Uploader Key Reset
```python
st.session_state.uploader_key += 1
st.rerun()
```

**Benefits**:
- Clears file uploader after successful submission
- Prevents accidental re-uploads
- Clean UX

---

## Testing Strategy

### v3.7.4
**Difficult to test**:
- All logic requires Streamlit runtime
- Cannot test stream parsing independently
- Cannot test tool invocation without UI
- Integration tests only

### v3.7.5
**Testable components**:

#### Unit Tests
```python
# Test StreamProcessor
def test_stream_processor_text_only():
    mock_stream = [
        {'messageStart': {'role': 'assistant'}},
        {'contentBlockStart': {'start': {}}},
        {'contentBlockDelta': {'delta': {'text': 'Hello'}}},
        {'contentBlockDelta': {'delta': {'text': ' world'}}},
        {'contentBlockStop': {}},
        {'messageStop': {'stopReason': 'end_turn'}},
        {'metadata': {'usage': {'inputTokens': 10, 'outputTokens': 5}}},
    ]
    
    result = process_stream(mock_stream)
    
    assert result.text == "Hello world"
    assert result.stop_reason == "end_turn"
    assert not result.has_tool_call
    assert result.metrics.input_tokens == 10

# Test ToolRegistry
def test_tool_registry_invoke():
    calculator = CalculatorBedrockConverseTool()
    registry = ToolRegistry([calculator])
    
    result = registry.invoke("calculator", {"expression": "2+2"})
    
    assert result == 4

# Test ConversationManager (with mock client)
def test_conversation_manager_tool_loop():
    mock_client = MockBedrockClient()  # Returns canned responses
    registry = ToolRegistry([...])
    
    manager = ConversationManager(
        bedrock_client=mock_client,
        tool_registry=registry,
        model_id="test-model",
        inference_config={},
        system_prompts=[],
    )
    
    result = manager.run([{"role": "user", "content": [{"text": "test"}]}])
    
    assert result.text == "expected response"
    assert mock_client.call_count == 2  # Initial + tool loop
```

#### Integration Tests
```python
# Test with real Bedrock (requires credentials)
def test_real_conversation():
    client = boto3.client('bedrock-runtime')
    registry = ToolRegistry([CalculatorBedrockConverseTool()])
    
    manager = ConversationManager(...)
    result = manager.run([{"role": "user", "content": [{"text": "What is 2+2?"}]}])
    
    assert "4" in result.text
```

---

## Migration Guide: v3.7.4 → v3.7.5

### For Developers Adding Features

#### Adding a New Tool

**v3.7.4**:
```python
# 1. Create tool class
class MyNewTool(BedrockConverseTool):
    ...

# 2. Instantiate in main file
my_new_tool = MyNewTool()

# 3. Add to tools list
tools = [calculator_tool, acronym_tool, ..., my_new_tool]

# 4. Add invocation logic in main file
for tool in tools:
    if tool.matches(tool_name):
        result = tool.invoke(...)  # Must know tool's signature
        break

# 5. Manually update system prompt to mention new tool
```

**v3.7.5**:
```python
# 1. Create tool class in cmn/tools/tool/
class MyNewTool(BedrockConverseTool):
    ...

# 2. Export from cmn/tools/tool/__init__.py
from .bedrock_converse_tools_tool_mynew import MyNewTool

# 3. Add to registry in main file (ONE LINE)
@st.cache_resource
def get_tool_registry():
    return ToolRegistry([
        ...,
        MyNewTool(),  # ← Add here
    ])

# Done! Tool is automatically:
# - Available for invocation (generic registry.invoke())
# - Included in tool_config
# - Documented in system prompt
```

#### Adding a Custom Renderer

**v3.7.5 only** (v3.7.4 doesn't support custom rendering):
```python
# 1. Create renderer class in cmn/tools/renderer/
class MyToolRenderer(BedrockConverseToolRenderer):
    def __init__(self):
        super().__init__(tool_name="my_tool")
    
    def render(self, tool_args: dict, tool_result: Any, container):
        # Custom Streamlit rendering
        container.markdown(f"**My Tool Result**")
        container.json(tool_result)

# 2. Export from cmn/tools/renderer/__init__.py
from .bedrock_converse_tools_renderer_mytool import MyToolRenderer

# 3. Add to registry in main file
@st.cache_resource
def get_renderer_registry():
    return RendererRegistry([
        ...,
        MyToolRenderer(),  # ← Add here
    ])

# Done! Tool results automatically render with custom logic
```

---

## Common Issues and Solutions

### Issue: "AttributeError: 'str' object has no attribute 'get'"

**Location**: 3_7_5_converse_tool_v3.py:671

**Cause**: Message content format inconsistency in session state

**Solution** (already fixed):
```python
# Before (caused error)
text = contents[0].get("text", "")

# After (handles both dict and string)
if isinstance(contents[0], dict):
    text = contents[0].get("text", "")
else:
    text = str(contents[0]) if contents else ""
```

### Issue: "W T W" mysterious messages appearing

**Cause**: Inconsistent text rendering methods (`.write()` vs `.markdown()`)

**Solution** (already fixed):
```python
# Before
st.chat_message("user").write(prompt)  # ← .write() can behave unexpectedly

# After
st.chat_message("user").markdown(prompt)  # ← Consistent with other rendering
```

### Issue: Tool calls not working in multi-turn conversations

**Cause**: v3.7.4 limitation (single tool call only)

**Solution**: Use v3.7.5 with ConversationManager loop

### Issue: Metrics show only last call, not full turn

**Cause**: v3.7.4 doesn't accumulate metrics

**Solution**: Use v3.7.5 InvocationStat with accumulate()

---

## Best Practices

### 1. Always Use v3.7.5 for Production
- Full multi-tool support
- Better error handling
- Persistent metrics
- Rich rendering

### 2. Tool Development
- Keep tools focused (single responsibility)
- Return structured data (dicts, not strings)
- Include comprehensive descriptions
- Handle errors gracefully (return error dict, don't raise)

### 3. Renderer Development
- Check result structure before rendering
- Fall back gracefully if unexpected format
- Use container parameter (don't call st.* directly)
- Keep rendering logic simple

### 4. System Prompts
- Let ToolRegistry generate tool documentation
- Add constraints (e.g., "call tools ONE AT A TIME")
- Don't duplicate tool descriptions manually

### 5. Session State
- Keep messages and stats arrays parallel
- Trim history regularly (MAX_MESSAGES)
- Reset uploader after successful submission

### 6. Performance
- Use `@st.cache_resource` for expensive objects
- Avoid recreating clients/registries
- Stream text deltas for responsive UI

---

## Future Enhancements

### Potential Improvements

1. **Parallel Tool Execution**
   - Currently tools execute sequentially
   - Could execute independent tools in parallel
   - Requires careful concurrency handling

2. **Tool Result Caching**
   - Cache expensive tool results (e.g., database queries)
   - Key: tool name + arguments hash
   - Invalidation strategy needed

3. **Streaming Tool Results**
   - Some tools (e.g., search) could stream results
   - Would require tool interface changes
   - Better UX for slow tools

4. **Tool Usage Analytics**
   - Track tool popularity
   - Measure tool latency
   - Identify failing tools
   - Store in database for analysis

5. **Dynamic Tool Loading**
   - Load tools from config file
   - Enable/disable tools per user
   - A/B test different tool sets

6. **Tool Composition**
   - Allow tools to call other tools
   - Build tool chains/workflows
   - Requires careful cycle detection

7. **Conversation Branching**
   - Fork conversation at any point
   - Compare different tool strategies
   - Undo/redo functionality

8. **Memory/Context Management**
   - Semantic compression of history
   - Relevance-based message pruning
   - External context store (vector DB)

---

## Conclusion

**v3.7.5 represents a complete architectural refactoring** that addresses all limitations of v3.7.4:

| Aspect | v3.7.4 | v3.7.5 |
|--------|--------|--------|
| Tool Loops | Single tool, single followup | Full multi-turn loops |
| Architecture | Monolithic, inline | Layered, modular |
| Testability | Requires Streamlit runtime | Pure functions, unit testable |
| Metrics | Per-call only | Cumulative per-turn |
| Tool Management | Manual list + iteration | ToolRegistry pattern |
| Rendering | JSON dumps only | Custom renderers |
| Error Handling | Inline, mixed with UI | Structured, separated |
| System Prompt | Static | Dynamic from registry |
| Reusability | Streamlit only | UI-agnostic core |
| Maintainability | Low (tight coupling) | High (loose coupling) |

**When to use each**:
- **v3.7.4**: Quick prototype, learning, simple single-tool scenarios
- **v3.7.5**: Production, complex workflows, multi-tool chains, maintainable codebase

**Migration**: Always prefer v3.7.5 for new development. Only maintain v3.7.4 as a reference/demo.

---

## Appendix: Code Organization

### v3.7.4 File Structure
```
pages/3_7_4_converse_tool_demo.py  (481 lines, everything inline)
```

### v3.7.5 File Structure
```
pages/3_7_5_converse_tool_v3.py                          (868 lines, UI layer)
cmn/bedrock/converse/conversation_manager.py             (150 lines, orchestration)
cmn/bedrock/converse/stream_processor.py                 (200 lines, parsing)
cmn/bedrock/converse/models.py                           (100 lines, data structures)
cmn/tools/tool/__init__.py                               (exports)
cmn/tools/tool/bedrock_converse_tools_tool.py           (base class)
cmn/tools/tool/bedrock_converse_tools_tool_*.py         (individual tools)
cmn/tools/renderer/__init__.py                           (exports)
cmn/tools/renderer/bedrock_converse_tools_renderer.py   (base class)
cmn/tools/renderer/bedrock_converse_tools_renderer_*.py (individual renderers)
```

**Total**: ~2000 lines spread across focused modules vs 481 lines of tangled code

---

*Document Version: 1.0*  
*Last Updated: 2026-05-31*  
*Author: System Architecture Documentation*
