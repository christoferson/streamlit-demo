"""
cmn/tools/tool/bedrock_converse_tools_tool_web_search.py
=========================================================
Tool: web_search

Client-side web search using DuckDuckGo (no API key required).

Note: AWS Bedrock does not support Anthropic's server-side web_search
tool (web_search_20260209 / web_search_20250305) — server tools are a
first-party Claude API feature. This tool provides equivalent capability
client-side via the standard tool-use loop.

Dependencies: ddgs (successor of duckduckgo_search)
"""

import logging
from datetime import datetime

from cmn.tools.tool.bedrock_converse_tools_tool import AbstractBedrockConverseTool

logger = logging.getLogger(__name__)

try:
    from ddgs import DDGS
except ImportError:  # older package name
    from duckduckgo_search import DDGS

_DEFAULT_MAX_RESULTS = 5
_MAX_SNIPPET_LENGTH = 1_000

# ddgs timelimit codes: d=past day, w=past week, m=past month, y=past year
_VALID_RECENCY = {"day": "d", "week": "w", "month": "m", "year": "y"}


class WebSearchBedrockConverseTool(AbstractBedrockConverseTool):

    def __init__(self, max_results: int = _DEFAULT_MAX_RESULTS):
        self.max_results = max_results
        name = "web_search"
        definition = {
            "toolSpec": {
                "name": name,
                "description": (
                    "Search the web for current information. "
                    "Use this when the answer depends on information not in the "
                    "conversation or likely past your knowledge cutoff: recent "
                    "events, current prices, latest versions, news, or anything "
                    "time-sensitive. Set search_type='news' for current events "
                    "and recency to restrict how old results may be. Results "
                    "include publication dates when available — check them "
                    "against today's date and discard stale items. Follow up "
                    "with url_content_loader to read a specific result in full."
                ),
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type":        "string",
                                "description": (
                                    "The web search query. "
                                    "Example: AWS Bedrock pricing 2026"
                                ),
                            },
                            "search_type": {
                                "type":        "string",
                                "enum":        ["web", "news"],
                                "description": (
                                    "'news' returns dated articles from news "
                                    "sources — use it for current events. "
                                    "'web' (default) is a general web search."
                                ),
                            },
                            "recency": {
                                "type":        "string",
                                "enum":        ["day", "week", "month", "year"],
                                "description": (
                                    "Only return results published within this "
                                    "window. Use 'day' or 'week' for news."
                                ),
                            },
                            "max_results": {
                                "type":        "integer",
                                "description": (
                                    f"Maximum number of results to return "
                                    f"(default {_DEFAULT_MAX_RESULTS}, max 10)."
                                ),
                            },
                        },
                        "required": ["query"],
                    }
                },
            }
        }
        super().__init__(name, definition)

    def summary(self) -> str:
        return "web_search : search the web for current or time-sensitive information"

    def invoke(self, params, tool_args=None):
        args  = tool_args or {}
        query = args.get("query", "").strip()

        if not query:
            return "No search query provided. Pass 'query' with a search term."

        max_results = min(int(args.get("max_results", self.max_results)), 10)
        search_type = args.get("search_type", "web")
        timelimit   = _VALID_RECENCY.get(args.get("recency", ""))

        logger.info(
            "WebSearchTool: query=%s type=%s recency=%s max_results=%d",
            query, search_type, timelimit, max_results,
        )

        try:
            ddgs = DDGS()
            if search_type == "news":
                results = ddgs.news(query, timelimit=timelimit, max_results=max_results)
            else:
                results = ddgs.text(query, timelimit=timelimit, max_results=max_results)

            if not results:
                return f"No web search results found for: {query}"

            today = datetime.now().strftime("%Y-%m-%d")
            formatted = []
            for i, r in enumerate(results, start=1):
                title   = r.get("title", "")
                # news results use 'url'/'date', text results use 'href'
                url     = r.get("href") or r.get("url", "")
                date    = r.get("date", "")
                snippet = r.get("body", "")[:_MAX_SNIPPET_LENGTH]
                date_line = f"   Published: {date}\n" if date else ""
                formatted.append(
                    f"{i}. {title}\n"
                    f"{date_line}"
                    f"   URL: {url}\n"
                    f"   {snippet}"
                )

            return (
                f"Web search results for '{query}' (retrieved {today}):\n\n"
                + "\n\n".join(formatted)
            )

        except Exception as e:
            logger.error("WebSearchTool: search failed query=%s error=%s", query, e)
            return f"Error performing web search: {str(e)}"
