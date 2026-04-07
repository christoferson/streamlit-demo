# cmn/bedrock_converse_tools_wikipedia.py

import wikipedia
from cmn.tools.tool.bedrock_converse_tools_tool import AbstractBedrockConverseTool

@DeprecationWarning
class WikipediaBedrockConverseTool(AbstractBedrockConverseTool):

    def __init__(self):
        name = "wikipedia_loader"
        definition = {
            "toolSpec": {
                "name": name,
                "description": (
                    "Useful for loading content from Wikipedia. "
                    "Use this when there is a need to fetch factual content from Wikipedia. "
                    "Accepts a search query and returns the summary and content "
                    "of the most relevant Wikipedia article."
                ),
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "expression": {
                                "type":        "string",
                                "description": (
                                    "The search query to look up on Wikipedia. "
                                    "Example: Melbourne climate"
                                ),
                            }
                        },
                        "required": ["expression"],
                    }
                },
            }
        }
        super().__init__(name, definition)

    def summary(self) -> str:
        return "wikipedia_loader : look up factual information from Wikipedia"

    def invoke(self, params, tool_args=None):
        args         = tool_args or {}
        search_query = args.get("expression", "").strip()   # ← fix: read from tool_args

        if not search_query:
            return "No search query provided. Pass 'expression' with a search term."

        try:
            search_results = wikipedia.search(search_query, results=3)

            if not search_results:
                return f"No Wikipedia results found for: {search_query}"

            for result in search_results:
                try:
                    page    = wikipedia.page(result, auto_suggest=False)
                    content = (
                        f"Title: {page.title}\n"
                        f"URL: {page.url}\n\n"
                        f"Summary:\n{page.summary}\n\n"
                        f"Full Content:\n{page.content[:8000]}"
                    )
                    return content

                except wikipedia.DisambiguationError as e:
                    try:
                        page    = wikipedia.page(e.options[0], auto_suggest=False)
                        content = (
                            f"Title: {page.title}\n"
                            f"URL: {page.url}\n\n"
                            f"Summary:\n{page.summary}\n\n"
                            f"Full Content:\n{page.content[:8000]}"
                        )
                        return content
                    except Exception:
                        continue

                except wikipedia.PageError:
                    continue

            return f"Could not retrieve a valid Wikipedia page for: {search_query}"

        except Exception as e:
            return f"Error fetching Wikipedia content: {str(e)}"