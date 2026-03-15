from cmn.bedrock_converse_tools import AbstractBedrockConverseTool
import requests
from bs4 import BeautifulSoup


class UrlContentBedrockConverseTool(AbstractBedrockConverseTool):

    def __init__(self):
        name = "url_content_loader"
        definition = {
            "toolSpec": {
                "name": name,
                "description": """Useful for loading content from a given URL. 
                This tool should only be used if there is a need to fetch and read the content of a specific web page.
                The tool accepts a URL and returns the text content of that page.""",
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "expression": {
                                "type": "string",
                                "description": "The URL of the web page to load. Example: https://example.com"
                            }
                        },
                        "required": [
                            "expression"
                        ]
                    }
                }
            }
        }
        super().__init__(name, definition)

    def invoke(self, params):
        try:
            target_url = params
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            }
            response = requests.get(target_url, headers=headers, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")

            # Remove script, style, nav, footer, and other non-content tags
            for tag in soup(["script", "style", "nav", "footer", "header", "aside", "meta", "link"]):
                tag.decompose()

            # Extract text from the body
            body = soup.find("body")
            if body:
                # Get text with whitespace cleanup
                lines = (line.strip() for line in body.get_text(separator="\n").splitlines())
                text = "\n".join(line for line in lines if line)
            else:
                lines = (line.strip() for line in soup.get_text(separator="\n").splitlines())
                text = "\n".join(line for line in lines if line)

            # Cap at 10000 characters to avoid token limits
            return text[:10000]

        except requests.RequestException as e:
            return f"Error fetching URL content: {str(e)}"