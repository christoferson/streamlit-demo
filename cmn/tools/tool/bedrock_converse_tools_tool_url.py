"""
cmn/tools/tool/bedrock_converse_tools_url.py
=============================================
Tool: url_content_loader

Fetches and extracts readable text content from a given URL.
Strips non-content tags (script, style, nav, footer, etc.)
and caps output at 10,000 characters to avoid token limits.

Dependencies: requests, beautifulsoup4
"""

import logging
import requests
from bs4 import BeautifulSoup

from cmn.tools.tool.bedrock_converse_tools_tool import AbstractBedrockConverseTool

logger = logging.getLogger(__name__)

_MAX_CONTENT_LENGTH = 10_000

_STRIP_TAGS = [
    "script", "style", "nav", "footer",
    "header", "aside", "meta", "link",
]

_REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

_REQUEST_TIMEOUT = 10


class UrlContentBedrockConverseTool(AbstractBedrockConverseTool):

    def __init__(self):
        name = "url_content_loader"
        definition = {
            "toolSpec": {
                "name": name,
                "description": (
                    "Useful for loading content from a given URL. "
                    "This tool should only be used if there is a need to fetch "
                    "and read the content of a specific web page. "
                    "The tool accepts a URL and returns the text content of that page."
                ),
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "expression": {
                                "type":        "string",
                                "description": "The URL of the web page to load. Example: https://example.com",
                            }
                        },
                        "required": ["expression"],
                    }
                },
            }
        }
        super().__init__(name, definition)

    def summary(self) -> str:
        return "url_content_loader : fetch and read content from a URL"

    def invoke(self, params, tool_args=None):
        args       = tool_args or {}
        target_url = args.get("expression", "").strip()

        if not target_url:
            return "No URL provided. Pass 'expression' with a valid URL."

        logger.info("UrlContentTool: fetching url=%s", target_url)

        try:
            response = requests.get(
                target_url,
                headers = _REQUEST_HEADERS,
                timeout = _REQUEST_TIMEOUT,
            )
            response.raise_for_status()

            return self._extract_text(response.text)

        except requests.RequestException as e:
            logger.error("UrlContentTool: request failed url=%s error=%s", target_url, e)
            return f"Error fetching URL content: {str(e)}"

    # ── Private ───────────────────────────────────────────────────────────────

    def _extract_text(self, html: str) -> str:
        """
        Parse HTML, strip non-content tags, extract clean text.
        Caps output at _MAX_CONTENT_LENGTH characters.
        """
        soup = BeautifulSoup(html, "html.parser")

        for tag in soup(_STRIP_TAGS):
            tag.decompose()

        body = soup.find("body") or soup
        lines = (
            line.strip()
            for line in body.get_text(separator="\n").splitlines()
        )
        text = "\n".join(line for line in lines if line)

        return text[:_MAX_CONTENT_LENGTH]