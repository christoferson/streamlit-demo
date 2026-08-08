import streamlit as st
import json
import logging
import cmn_settings
import base64
from typing import Optional, Callable, Any
from dataclasses import dataclass, field

from anthropic import AnthropicBedrockMantle

from openai import OpenAI

from cmn.view.mime_constants import mime_mapping_image, mime_mapping_document
from cmn.view import CONVERSE_TOOL_GUIDE

from cmn.tools.tool import (
    UrlContentBedrockConverseTool,
    WebSearchBedrockConverseTool,
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
# SECTION: OpenAI (Mantle) Conversation Manager
################################################################################

_RESPONSE_TERMINAL_EVENTS = frozenset({
    "response.completed",
    "response.incomplete",   # hit max_output_tokens / content filter
    "response.failed",
})


class OpenAIMantleConversationManager:
    """
    Orchestrates multi-turn conversations for OpenAI models served on the
    Bedrock Mantle /openai/v1 endpoint. GPT-5.5 on Mantle requires the
    Responses API (chat completions is rejected with a 400).
    Handles tool-use loops with streaming support — same callback contract
    as MantleConversationManager, plus two extra callbacks for Bedrock's
    server-side web_search tool (retrieval steps and url citations).
    """

    def __init__(
        self,
        openai_client: OpenAI,
        tool_registry: Optional[ToolRegistry],
        model_id: str,
        max_tokens: int,
        temperature: float,
        system_prompt: str,
        native_web_search: bool = False,
        external_web_access: bool = False,
    ):
        self.client = openai_client
        self.registry = tool_registry
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.system_prompt = system_prompt
        self.native_web_search = native_web_search
        self.external_web_access = external_web_access

    def run(
        self,
        message_history: list,
        on_text_delta: Optional[Callable[[str], None]] = None,
        on_message_complete: Optional[Callable[[dict, int], None]] = None,
        on_tool_invoked: Optional[Callable[[str, dict, Any], None]] = None,
        on_builtin_search: Optional[Callable[[str, dict], None]] = None,
        on_citations: Optional[Callable[[list], None]] = None,
    ) -> tuple[str, dict]:
        """
        Run conversation with tool loop support.
        Returns: (final_text, usage_dict)
        """
        input_items = self._convert_messages(message_history)
        accumulated_text = ""
        total_usage = {'input_tokens': 0, 'output_tokens': 0}

        while True:
            import time
            start_time = time.time()

            api_params = {
                "model": self.model_id,
                "max_output_tokens": self.max_tokens,
                "instructions": self.system_prompt,
                "input": input_items,
                "stream": True,
            }

            tools = self._build_tools()
            if tools:
                api_params["tools"] = tools

            stream = self.client.responses.create(**api_params)

            function_calls = []
            final_response = None

            for event in stream:
                if event.type == "response.output_text.delta":
                    accumulated_text += event.delta
                    if on_text_delta:
                        on_text_delta(event.delta)
                elif event.type in _RESPONSE_TERMINAL_EVENTS:
                    # A run that hits max_output_tokens ends in
                    # response.incomplete, not response.completed — take the
                    # response off whichever terminal event arrives, or usage,
                    # citations and tool calls are all silently lost.
                    final_response = event.response

            latency_ms = int((time.time() - start_time) * 1000)

            if final_response is not None:
                usage = final_response.usage
                if usage:
                    total_usage['input_tokens'] += usage.input_tokens or 0
                    total_usage['output_tokens'] += usage.output_tokens or 0
                function_calls = [
                    item for item in final_response.output
                    if item.type == "function_call"
                ]
                self._report_builtin_search(final_response, on_builtin_search)
                self._report_citations(final_response, on_citations)

            if on_message_complete:
                on_message_complete(total_usage, latency_ms)

            if not function_calls or not self.registry:
                return accumulated_text, total_usage

            # Echo the model's output items, then append one
            # function_call_output per call
            for item in final_response.output:
                input_items.append(item)

            for fc in function_calls:
                tool_name = fc.name
                try:
                    tool_input = json.loads(fc.arguments) if fc.arguments else {}
                except json.JSONDecodeError:
                    tool_input = {}

                try:
                    result = self.registry.invoke(tool_name, tool_input)

                    if on_tool_invoked:
                        on_tool_invoked(tool_name, tool_input, result)

                    output = str(result)
                except Exception as e:
                    logger.error(f"Tool {tool_name} failed: {e}")
                    output = f"Error: {str(e)}"

                input_items.append({
                    "type": "function_call_output",
                    "call_id": fc.call_id,
                    "output": output,
                })

    def _build_tools(self) -> list:
        """
        Responses API uses a flat function-tool format. Bedrock's built-in
        web_search is a server-side tool — it carries no schema and is not
        dispatched through the registry; Bedrock runs the retrieval itself
        and returns the answer with url citations.
        """
        tools = []

        if self.native_web_search:
            tools.append({
                "type": "web_search",
                # Live-web retrieval additionally requires the
                # bedrock-websearch:ExternalWebAccess IAM permission; the
                # Amazon-operated index needs only InvokeSearch.
                "external_web_access": self.external_web_access,
            })

        if self.registry and hasattr(self.registry, 'get_openai_tools'):
            tools.extend(
                {
                    "type": "function",
                    "name":        t["function"]["name"],
                    "description": t["function"]["description"],
                    "parameters":  t["function"]["parameters"],
                }
                for t in self.registry.get_openai_tools()
            )

        return tools

    @staticmethod
    def _report_builtin_search(response, on_builtin_search) -> None:
        """
        Surface each server-side retrieval step. Bedrock emits one
        web_search_call item per step: action.type 'search' carries the
        queries it formulated, 'open_page' the URL it read in full, and
        'find_in_page' a pattern it looked for inside a loaded page.
        """
        if not on_builtin_search:
            return

        for item in response.output:
            if getattr(item, "type", None) != "web_search_call":
                continue
            action = getattr(item, "action", None)
            action_type = getattr(action, "type", "search")
            detail = {}
            if action_type == "search":
                # Bedrock returns 'queries'; the OpenAI SDK's own search
                # action model uses singular 'query' — accept either.
                queries = list(getattr(action, "queries", None) or [])
                if not queries:
                    single = getattr(action, "query", None)
                    if single:
                        queries = [single]
                detail["queries"] = queries
            elif action_type == "open_page":
                detail["url"] = getattr(action, "url", "")
            elif action_type == "find_in_page":
                detail["url"]     = getattr(action, "url", "")
                detail["pattern"] = getattr(action, "pattern", "")
            on_builtin_search(action_type, detail)

    @staticmethod
    def _report_citations(response, on_citations) -> None:
        """
        Collect url_citation annotations attached to the output text.
        Each has title/url plus start_index/end_index offsets into the text.
        """
        if not on_citations:
            return

        citations = []
        for item in response.output:
            if getattr(item, "type", None) != "message":
                continue
            for content in getattr(item, "content", None) or []:
                if getattr(content, "type", None) != "output_text":
                    continue
                for ann in getattr(content, "annotations", None) or []:
                    if getattr(ann, "type", None) != "url_citation":
                        continue
                    citations.append({
                        "title": getattr(ann, "title", "") or "",
                        "url":   getattr(ann, "url", "") or "",
                    })

        if citations:
            on_citations(citations)

    def _convert_messages(self, message_history: list) -> list:
        """Convert internal message format to Responses API input items."""
        converted = []
        for msg in message_history:
            role = msg["role"]
            content = msg["content"]

            if len(content) == 1 and "text" in content[0]:
                converted.append({"role": role, "content": content[0]["text"]})
            else:
                text_type = "output_text" if role == "assistant" else "input_text"
                openai_content = []
                for item in content:
                    if "text" in item:
                        openai_content.append({
                            "type": text_type,
                            "text": item["text"],
                        })
                    elif "image" in item:
                        img_data = item["image"]
                        b64 = base64.b64encode(
                            img_data['source']['bytes']
                        ).decode('utf-8')
                        openai_content.append({
                            "type": "input_image",
                            "image_url": f"data:image/{img_data['format']};base64,{b64}",
                        })
                    elif "document" in item:
                        doc_data = item["document"]
                        openai_content.append({
                            "type": text_type,
                            "text": f"[Document: {doc_data.get('name', 'unknown')}]",
                        })
                converted.append({"role": role, "content": openai_content})

        return converted


################################################################################
# SECTION: File / Media Utilities
################################################################################

def build_user_message(prompt: str, files=None) -> dict:
    """
    Construct the user content list from the chat_input submission,
    attaching any images/documents in the message they arrived with.
    """
    content = [{"text": prompt}]

    for f in files or []:
        if f.type in mime_mapping_image:
            content.append({
                "image": {
                    "format": mime_mapping_image[f.type],
                    "source": {"bytes": f.getvalue()},
                }
            })
        elif f.type in mime_mapping_document:
            content.append({
                "document": {
                    "format": mime_mapping_document[f.type],
                    "name":   f.name.replace(".", "_").replace(" ", "_"),
                    "source": {"bytes": f.getvalue()},
                }
            })

    return {"role": "user", "content": content}


################################################################################
# SECTION: Shared Streamlit Resources
################################################################################

@st.cache_resource
def get_mantle_client():
    return AnthropicBedrockMantle(aws_region=AWS_REGION)


@st.cache_resource(ttl=3300)  # refresh before the 1h bearer token expires
def get_openai_mantle_client():
    """
    OpenAI models on Bedrock Mantle are served via the OpenAI-compatible
    endpoint (/openai/v1). The OpenAI SDK cannot SigV4-sign requests, so a
    short-lived bearer token is derived from the IAM credential chain via
    aws-bedrock-token-generator. The cache TTL (55 min) rotates the token
    before its 1h expiry.
    """
    from datetime import timedelta
    from aws_bedrock_token_generator import provide_token

    token = provide_token(region=AWS_REGION, expiry=timedelta(hours=1))
    return OpenAI(
        base_url=f"https://bedrock-mantle.{AWS_REGION}.api.aws/openai/v1",
        api_key=token,
    )


def is_openai_model(model_id: str) -> bool:
    return model_id.startswith("openai.")


################################################################################
# SECTION: Web Search Mode
################################################################################

# Bedrock's built-in web_search is a server-side tool on the Mantle Responses
# API, so it is only available to the openai.* models. Anthropic models keep
# using the client-side DuckDuckGo tool loop.
WEB_SEARCH_OFF      = "Off"
WEB_SEARCH_CLIENT   = "Client-side (DuckDuckGo)"
WEB_SEARCH_BEDROCK  = "Bedrock built-in (server-side)"

WEB_SEARCH_MODES = [WEB_SEARCH_OFF, WEB_SEARCH_CLIENT, WEB_SEARCH_BEDROCK]

WEB_SEARCH_HELP = (
    "**Off** — no search tools.\n\n"
    "**Client-side** — adds the `web_search` and `url_content_loader` tools, "
    "run locally via DuckDuckGo through the normal tool-use loop. Works with "
    "every model here.\n\n"
    "**Bedrock built-in** — Bedrock performs retrieval server-side against "
    "Amazon's web index and returns the answer with url citations. "
    "OpenAI models only (Responses API); requires the "
    "`bedrock-websearch:InvokeSearch` IAM permission."
)


@st.cache_resource
def get_tool_registry(enable_web_search: bool):
    """
    Build the tool registry based on sidebar options.
    Returns None when no tools are enabled (basic chat mode).
    """
    tools = []

    if enable_web_search:
        tools.append(WebSearchBedrockConverseTool())
        tools.append(UrlContentBedrockConverseTool())

    if not tools:
        return None

    return ToolRegistry(tools)
    # Other available tools:
    #     CalculatorBedrockConverseTool(),
    #     AcronymBedrockConverseTool(),
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


opt_model_id_list = [
    #"global.anthropic.claude-opus-4-7",
    "anthropic.claude-opus-4-7",
    "openai.gpt-5.5",
    "openai.gpt-5.6-sol",
    "openai.gpt-5.6-terra",
    "openai.gpt-5.6-luna",
    #"global.anthropic.claude-sonnet-4-20250514-v1:0",
    #"anthropic.claude-sonnet-4-20250514-v1:0",
    #"anthropic.claude-opus-4-6",
    #"anthropic.claude-sonnet-4-6",
    #"anthropic.claude-sonnet-4-5",
    #"anthropic.claude-mythos-preview",
]

OPENAI_MODEL_IDS = [m for m in opt_model_id_list if is_openai_model(m)]

DEFAULT_MODEL_ID = "openai.gpt-5.6-luna"

# Looked up rather than hardcoded so commenting models in/out above cannot
# silently shift the default to a different entry.
DEFAULT_MODEL_INDEX = (
    opt_model_id_list.index(DEFAULT_MODEL_ID)
    if DEFAULT_MODEL_ID in opt_model_id_list else 0
)

with st.sidebar:
    opt_model_id = st.selectbox(
        "Model ID",
        opt_model_id_list,
        index=DEFAULT_MODEL_INDEX,
        key="bedrock_mantle_model_id"
    )

    opt_web_search_mode = st.radio(
        "Web Search",
        WEB_SEARCH_MODES,
        index=1,
        key="bedrock_mantle_web_search_mode",
        help=WEB_SEARCH_HELP,
    )

    # Falling back keeps the request valid rather than erroring at call time.
    if opt_web_search_mode == WEB_SEARCH_BEDROCK and not is_openai_model(opt_model_id):
        st.warning(
            f"**Using {WEB_SEARCH_CLIENT} instead.**\n\n"
            f"`{opt_model_id}` is an Anthropic model, which Bedrock serves "
            f"through the Messages API. Built-in web search is a server-side "
            f"tool on the Responses API, so only the `openai.*` models can "
            f"use it.\n\n"
            f"To use it, switch **Model ID** to one of: "
            + ", ".join(f"`{m}`" for m in OPENAI_MODEL_IDS)
            + ".",
            icon=":material/travel_explore:",
        )
        opt_web_search_mode = WEB_SEARCH_CLIENT

    opt_external_web_access = False
    if opt_web_search_mode == WEB_SEARCH_BEDROCK:
        opt_external_web_access = st.checkbox(
            "Allow live web access",
            value=False,
            key="bedrock_mantle_external_web_access",
            help=(
                "Off: retrieve from Amazon's pre-indexed corpus only. "
                "On: allow Bedrock to fetch live pages — requires the "
                "`bedrock-websearch:ExternalWebAccess` IAM permission and "
                "fails with AccessDeniedException without it."
            ),
        )

opt_native_web_search = opt_web_search_mode == WEB_SEARCH_BEDROCK
opt_client_web_search = opt_web_search_mode == WEB_SEARCH_CLIENT
# Kept for the citation-rules block in the runtime system prompt, which
# applies to either search mode.
opt_web_search = opt_web_search_mode != WEB_SEARCH_OFF

mantle_client = get_mantle_client()
tool_registry = get_tool_registry(opt_client_web_search)


################################################################################
# SECTION: System Prompt
################################################################################

def build_default_system_prompt(
    registry: Optional[ToolRegistry],
    native_web_search: bool = False,
) -> str:
    base_prompt = "You are a helpful AI assistant."

    if native_web_search:
        # No tool summary — Bedrock runs retrieval server-side and the model
        # is not told about a callable search tool.
        return "\n\n".join([
            base_prompt,
            #"You have built-in web search. When the answer depends on current "
            #"or time-sensitive information (recent events, prices, versions, "
            #"news), search before answering rather than answering from memory. "
            #"Cite the sources you used.",
        ])

    if registry and hasattr(registry, 'build_tool_summary'):
        tool_summary = registry.build_tool_summary()
        search_guidance = None
        if "web_search" in registry.tool_names:
            search_guidance = (
                "When the answer depends on current or time-sensitive information "
                "(recent events, prices, versions, news), use web_search before "
                "answering rather than answering from memory. "
                "For news, use search_type='news' with recency='day' or 'week'. "
                "Check the Published date on each result against today's date "
                "and discard stale items — do not present old articles as "
                "current news. "
                "When the user asks for the latest news, ALWAYS run a fresh "
                "web_search — never recap news from earlier in this conversation, "
                "as it may already be outdated. "
                "Use url_content_loader to read a specific result in full."
            )
        return "\n\n".join(filter(None, [
            base_prompt,
            tool_summary,
            search_guidance,
            "Call tools ONE AT A TIME. Wait for each result before calling the next.",
        ]))

    return base_prompt


def build_runtime_system_prompt(
    user_system_msg: str,
    web_search_enabled: bool = False,
    native_web_search: bool = False,
) -> str:
    """
    Final system prompt sent to the API. Appends the current date (and, when
    web search is on, citation rules) at request time so they stay correct
    regardless of the (session-state-cached) sidebar text and across
    long-running server processes.
    """
    from datetime import datetime
    today = datetime.now().strftime("%A, %B %d, %Y")
    parts = [
        user_system_msg,
        (
            f"Today's date is {today}. Your training data has a cutoff in the "
            "past — treat anything you recall from memory as potentially outdated."
        ),
    ]
    if native_web_search:
        # Bedrock returns url_citation annotations separately; asking for
        # inline links keeps the answer text self-contained too.
        parts.append(
            "CITATION RULES: state the publication date of any time-sensitive "
            "claim, and link the source inline as markdown. Never present a "
            "claim from a search result without its source."
        )
    elif web_search_enabled:
        parts.append(
            "CITATION RULES for web search results:\n"
            "- Every news item or fact taken from a search result MUST include "
            "its publication date and a markdown link to the source, e.g.: "
            "**Headline** (2026-07-15) — [reuters.com](https://www.reuters.com/...)\n"
            "- If a result has no publication date, say 'date not stated' "
            "rather than omitting the date or guessing it.\n"
            "- Never present a claim from search results without its source link."
        )
    return "\n\n".join(parts)


OPT_SYSTEM_MSG_DEFAULT = build_default_system_prompt(
    tool_registry,
    native_web_search=opt_native_web_search,
)


def sync_system_msg_default(new_default: str) -> None:
    """
    The System Message text area is keyed, so once Streamlit has stored a value
    it ignores the `value` argument on later reruns — toggling Web Search would
    leave the AVAILABLE TOOLS / search guidance text describing tools the model
    no longer has (or omitting ones it now does).

    Re-seed the widget whenever the computed default changes, but only if the
    user has not edited it — a hand-written prompt is never overwritten.
    """
    key      = "bedrock_mantle_system_msg"
    prev_key = "bedrock_mantle_system_msg_default"
    prev     = st.session_state.get(prev_key)

    if prev == new_default:
        return

    st.session_state[prev_key] = new_default
    if key not in st.session_state or st.session_state[key] == prev:
        st.session_state[key] = new_default


sync_system_msg_default(OPT_SYSTEM_MSG_DEFAULT)


################################################################################
# SECTION: Streamlit Sidebar / Options
################################################################################

# Model ID and Web Search live in the "Web Search Mode" section above — the
# search options depend on which model is selected.

with st.sidebar:
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
    # Value comes from session state, seeded/refreshed by
    # sync_system_msg_default() — do not pass `value` here.
    opt_system_msg = st.text_area(
        "System Message",
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
    elif opt_native_web_search:
        st.info("Server-side web search — no client-side tools")
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

################################################################################
# SECTION: Render Message History
################################################################################

for idx, msg in enumerate(st.session_state.bedrock_mantle_messages):
    with st.chat_message(msg["role"]):
        contents = msg["content"]
        text = contents[0].get("text", "")

        st.markdown(text)
        if msg["role"] == "user":
            for extra in contents[1:]:
                if "document" in extra:
                    doc_name = extra["document"].get("name", "")
                    st.caption(f":green[Document: {doc_name}]")
                elif "image" in extra:
                    img_bytes = extra["image"]["source"].get("bytes")
                    if img_bytes:
                        st.image(img_bytes, width=200)

        if opt_show_metrics and msg["role"] == "assistant":
            stat = st.session_state.bedrock_mantle_invocation_stats[idx]
            if stat is not None:
                st.caption(stat.as_markdown())


################################################################################
# SECTION: Command Panel + Chat Input (pinned to viewport bottom)
################################################################################

with st.bottom:
    with st.container(
        horizontal=True,
        horizontal_alignment="right",
        vertical_alignment="center",
        border=False,
        gap="xxsmall",
    ):
        st.markdown(
            f":violet[**{len(st.session_state.bedrock_mantle_messages)}/{MAX_MESSAGES}**]",
            help="Messages in conversation history",
        )
        if st.button(
            ":material/delete_history:",
            type="tertiary",
            key="bedrock_mantle_clear_history",
            help="Clear conversation history",
        ):
            st.session_state.bedrock_mantle_messages = []
            st.session_state.bedrock_mantle_invocation_stats = []
            st.rerun()

    submission = st.chat_input(
        f"Ask {opt_model_id}",
        key="bedrock_mantle_chat_input",
        accept_file="multiple",
        file_type=["png", "jpg", "jpeg", "gif", "webp", "txt", "csv", "pdf", "md"],
    )

if submission and submission.text:
    prompt = submission.text

    # Build user message with any attachments
    user_message = build_user_message(prompt, files=submission.files)

    message_history = st.session_state.bedrock_mantle_messages.copy()
    message_history.append(user_message)

    with st.chat_message("user"):
        st.write(prompt)
        for f in submission.files or []:
            if f.type in mime_mapping_image:
                st.image(f.getvalue(), caption=f.name, width=300)
            else:
                st.caption(f":green[Document: {f.name}]")

    # Per-turn accumulators
    accumulated = {"text": ""}
    turn_stat = InvocationStat()

    with st.chat_message("assistant"):
        result_area = st.empty()

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

        # Server-side search steps and citations arrive with the completed
        # response, so they are appended after the streamed text rather than
        # interleaved. Collected here, rendered once the run finishes.
        search_steps = []
        citations = []

        def on_builtin_search(action_type: str, detail: dict):
            """Fired per Bedrock server-side retrieval step."""
            turn_stat.record_tool(f"web_search:{action_type}")
            if action_type == "search":
                for q in detail.get("queries", []):
                    search_steps.append(f"searched `{q}`")
            elif action_type == "open_page":
                search_steps.append(f"read <{detail.get('url', '')}>")
            elif action_type == "find_in_page":
                search_steps.append(
                    f"found `{detail.get('pattern', '')}` in "
                    f"<{detail.get('url', '')}>"
                )

        def on_citations(new_citations: list):
            """
            Fired with the url_citation annotations on the answer. The same
            URL is commonly cited for several spans of text, so dedupe as we
            go — including within a single batch.
            """
            seen = {c["url"] for c in citations}
            for c in new_citations:
                if c["url"] and c["url"] not in seen:
                    seen.add(c["url"])
                    citations.append(c)

        if is_openai_model(opt_model_id):
            manager = OpenAIMantleConversationManager(
                openai_client=get_openai_mantle_client(),
                tool_registry=tool_registry,
                model_id=opt_model_id,
                max_tokens=opt_max_tokens,
                temperature=0.0,
                system_prompt=build_runtime_system_prompt(
                    opt_system_msg,
                    web_search_enabled=opt_web_search,
                    native_web_search=opt_native_web_search,
                ),
                native_web_search=opt_native_web_search,
                external_web_access=opt_external_web_access,
            )
        else:
            manager = MantleConversationManager(
                mantle_client=mantle_client,
                tool_registry=tool_registry,
                model_id=opt_model_id,
                max_tokens=opt_max_tokens,
                temperature=0.0,
                system_prompt=build_runtime_system_prompt(
                    opt_system_msg,
                    web_search_enabled=opt_web_search,
                ),
            )

        # Return values unused — text/usage are collected via the callbacks
        # into `accumulated` and `turn_stat`
        with st.spinner("Processing...", show_time=True, width="content"):
            try:
                run_kwargs = {
                    "on_text_delta":       on_text_delta,
                    "on_message_complete": on_message_complete,
                    "on_tool_invoked":     on_tool_invoked,
                }
                if isinstance(manager, OpenAIMantleConversationManager):
                    run_kwargs["on_builtin_search"] = on_builtin_search
                    run_kwargs["on_citations"]      = on_citations

                manager.run(message_history, **run_kwargs)
            except Exception as e:
                st.error(f"Error: {str(e)}")
                logger.exception("Error in conversation manager")
                st.stop()

        # Append the server-side retrieval trace + sources so they persist in
        # the message history the same way client-side tool traces do.
        if search_steps:
            accumulated["text"] += (
                "\n\n:blue[🌐 **Bedrock web search**]  \n"
                + "  \n".join(f"- {s}" for s in search_steps)
            )
        if citations:
            accumulated["text"] += "\n\n**Sources**  \n" + "  \n".join(
                f"- [{c['title'] or c['url']}]({c['url']})" for c in citations
            )
        if search_steps or citations:
            result_area.markdown(accumulated["text"])

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