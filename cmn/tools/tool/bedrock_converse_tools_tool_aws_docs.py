# bedrock_converse_tools_tool_aws_docs.py

import httpx
import logging
from bs4 import BeautifulSoup
from cmn.tools.tool.bedrock_converse_tools_tool import AbstractBedrockConverseTool

logger = logging.getLogger(__name__)

# ── Known correct AWS docs URL patterns ──────────────────────────────────────
AWS_DOCS_BASE = "https://docs.aws.amazon.com"

KNOWN_DOCS = {
    "lambda":   "https://docs.aws.amazon.com/lambda/latest/dg",
    "s3":       "https://docs.aws.amazon.com/AmazonS3/latest/userguide",
    "dynamodb": "https://docs.aws.amazon.com/amazondynamodb/latest/developerguide",
    "ec2":      "https://docs.aws.amazon.com/AWSEC2/latest/UserGuide",
    "bedrock":  "https://docs.aws.amazon.com/bedrock/latest/userguide",
    "iam":      "https://docs.aws.amazon.com/IAM/latest/UserGuide",
    "rds":      "https://docs.aws.amazon.com/AmazonRDS/latest/UserGuide",
    "ecs":      "https://docs.aws.amazon.com/AmazonECS/latest/developerguide",
    "eks":      "https://docs.aws.amazon.com/eks/latest/userguide",
    "sqs":      "https://docs.aws.amazon.com/AWSSimpleQueueService/latest/SQSDeveloperGuide",
    "sns":      "https://docs.aws.amazon.com/sns/latest/dg",
    "cloudformation": "https://docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide",
}


class AwsDocsBedrockConverseTool(AbstractBedrockConverseTool):

    def __init__(self):
        name = "aws_docs"
        definition = {
            "toolSpec": {
                "name": name,
                "description": (
                    "Search and read AWS official documentation. "
                    "Use this to answer questions about AWS services, "
                    "APIs, pricing, best practices, and limits/quotas. "
                    "For action=read, use real AWS docs URLs only."
                ),
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "action": {
                                "type": "string",
                                "enum": ["search", "read"],
                                "description": (
                                    "search: find relevant AWS docs pages using Google. "
                                    "read: fetch content from a specific AWS docs URL."
                                ),
                            },
                            "query": {
                                "type": "string",
                                "description": "Search query. Example: 'Lambda concurrency limits'",
                            },
                            "url": {
                                "type": "string",
                                "description": (
                                    "Real AWS docs URL to read. "
                                    "Must start with https://docs.aws.amazon.com"
                                ),
                            },
                            "service": {
                                "type": "string",
                                "enum": list(KNOWN_DOCS.keys()),
                                "description": "AWS service to search within.",
                            },
                        },
                        "required": ["action"],
                    }
                },
            }
        }
        super().__init__(name, definition)

    def summary(self) -> str:
        return (
            "aws_docs : search and read AWS official documentation. "
            "Use for questions about AWS services, APIs, pricing, limits and best practices. "
            "First search to find the right URL, then read to get full content."
        )

    def invoke(self, params, tool_args: dict = None) -> dict:
        args    = tool_args or {}
        action  = args.get("action", "search")
        query   = args.get("query", "")
        url     = args.get("url", "")
        service = args.get("service", "")

        logger.info("AwsDocsTool: action=%s query=%s service=%s", action, query, service)

        if action == "search":
            if not query:
                return {"error": "query is required for search"}
            return self._search(query, service)

        if action == "read":
            if not url:
                return {"error": "url is required for read"}
            return self._read(url)

        return {"error": f"Unknown action: {action}"}

    # ── Search via Google site: search ────────────────────────────────────

    def _search(self, query: str, service: str = "") -> dict:
        """
        Search AWS docs using Google site: search.
        More reliable than AWS's own search endpoint.
        """
        try:
            # Build site-restricted Google search
            site = f"site:docs.aws.amazon.com"
            if service and service in KNOWN_DOCS:
                # Restrict to service subdirectory
                base = KNOWN_DOCS[service].replace("https://", "")
                site = f"site:{base}"

            search_query = f"{site} {query}"

            response = httpx.get(
                "https://www.google.com/search",
                params={"q": search_query, "num": 5},
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"
                    )
                },
                timeout=15,
                follow_redirects=True,
            )

            soup    = BeautifulSoup(response.text, "html.parser")
            results = []

            for g in soup.select("div.g")[:5]:
                link_el  = g.select_one("a[href]")
                title_el = g.select_one("h3")
                desc_el  = g.select_one(".VwiC3b, .s3v9rd, span")

                if not link_el or not title_el:
                    continue

                href = link_el["href"]
                if not href.startswith("https://docs.aws.amazon.com"):
                    continue

                results.append({
                    "title":       title_el.get_text(strip=True),
                    "url":         href,
                    "description": desc_el.get_text(strip=True)[:200] if desc_el else "",
                })

            if not results:
                # Last resort — return known service docs homepage
                return self._known_pages(query, service)

            return {
                "query":   query,
                "results": results,
                "hint":    "Call aws_docs with action=read and one of these URLs",
            }

        except Exception as e:
            logger.error("AwsDocsTool search error: %s", e)
            return self._known_pages(query, service)

    # ── Read ──────────────────────────────────────────────────────────────

    def _read(self, url: str) -> dict:
        """Fetch and parse an AWS docs page."""
        try:
            if not url.startswith("https://docs.aws.amazon.com"):
                return {"error": "Only docs.aws.amazon.com URLs are allowed"}

            response = httpx.get(
                url,
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=15,
                follow_redirects=True,
            )

            if response.status_code == 404:
                return {
                    "error": f"Page not found (404): {url}",
                    "hint":  "Try searching again with action=search to find the correct URL",
                }

            if response.status_code != 200:
                return {"error": f"HTTP {response.status_code}: {url}"}

            soup = BeautifulSoup(response.text, "html.parser")

            # Remove noise
            for tag in soup(["nav", "footer", "script", "style",
                              "header", ".feedback-container", ".prev-next"]):
                tag.decompose()

            # Get main content
            main = (
                soup.select_one("#main-content")
                or soup.select_one(".main-content")
                or soup.select_one("article")
                or soup.select_one("main")
                or soup.body
            )

            if not main:
                return {"error": "Could not extract page content"}

            text = main.get_text(separator="\n", strip=True)

            # Truncate
            max_chars = 8000
            if len(text) > max_chars:
                text = text[:max_chars] + "\n\n[Content truncated — page has more content]"

            title_el = soup.select_one("h1")
            title    = title_el.get_text(strip=True) if title_el else ""

            return {
                "url":     url,
                "title":   title,
                "content": text,
            }

        except Exception as e:
            logger.error("AwsDocsTool read error: %s", e)
            return {"error": str(e)}

    # ── Known pages fallback ──────────────────────────────────────────────

    def _known_pages(self, query: str, service: str) -> dict:
        """Return known correct AWS docs URLs as fallback."""

        # Hardcoded known pages for common queries
        known = {
            "lambda": [
                ("Lambda quotas",        "https://docs.aws.amazon.com/lambda/latest/dg/gettingstarted-limits.html"),
                ("Lambda concurrency",   "https://docs.aws.amazon.com/lambda/latest/dg/lambda-concurrency.html"),
                ("Lambda pricing",       "https://aws.amazon.com/lambda/pricing/"),
                ("Lambda overview",      "https://docs.aws.amazon.com/lambda/latest/dg/welcome.html"),
            ],
            "s3": [
                ("S3 quotas",            "https://docs.aws.amazon.com/AmazonS3/latest/userguide/BucketRestrictions.html"),
                ("S3 storage classes",   "https://docs.aws.amazon.com/AmazonS3/latest/userguide/storage-class-intro.html"),
            ],
            "dynamodb": [
                ("DynamoDB quotas",      "https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/ServiceQuotas.html"),
                ("DynamoDB pricing",     "https://aws.amazon.com/dynamodb/pricing/"),
            ],
            "bedrock": [
                ("Bedrock quotas",       "https://docs.aws.amazon.com/bedrock/latest/userguide/quotas.html"),
                ("Bedrock models",       "https://docs.aws.amazon.com/bedrock/latest/userguide/models-supported.html"),
            ],
        }

        results = []

        if service and service in known:
            for title, url in known[service]:
                results.append({"title": title, "url": url, "description": ""})
        else:
            # Generic fallback
            results.append({
                "title":       "AWS Documentation Search",
                "url":         f"https://docs.aws.amazon.com/search/doc-search.html?searchQuery={query.replace(' ', '+')}",
                "description": "Search AWS documentation directly",
            })

        return {
            "query":   query,
            "results": results,
            "hint":    "Call aws_docs with action=read and one of these URLs",
        }