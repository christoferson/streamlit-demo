from cmn.bedrock_converse_tools import AbstractBedrockConverseTool
import wikipedia


class WikipediaBedrockConverseTool(AbstractBedrockConverseTool):

    def __init__(self):
        name = "wikipedia_loader"
        definition = {
            "toolSpec": {
                "name": name,
                "description": """Useful for loading content from Wikipedia. 
                This tool should be used when there is a need to fetch and read factual content from Wikipedia.
                The tool accepts a search query and returns the summary and content of the most relevant Wikipedia article.""",
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "expression": {
                                "type": "string",
                                "description": "The search query to look up on Wikipedia. Example: Melbourne climate"
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

    def invoke(self, params, tool_args=None):
        try:
            search_query = params

            # Search for the most relevant page
            search_results = wikipedia.search(search_query, results=3)
            if not search_results:
                return f"No Wikipedia results found for: {search_query}"

            # Try each result until we get a valid page
            for result in search_results:
                try:
                    page = wikipedia.page(result, auto_suggest=False)
                    content = f"Title: {page.title}\n"
                    content += f"URL: {page.url}\n\n"
                    content += f"Summary:\n{page.summary}\n\n"
                    content += f"Full Content:\n{page.content[:8000]}"
                    return content
                except wikipedia.DisambiguationError as e:
                    # If disambiguation, try the first option
                    try:
                        page = wikipedia.page(e.options[0], auto_suggest=False)
                        content = f"Title: {page.title}\n"
                        content += f"URL: {page.url}\n\n"
                        content += f"Summary:\n{page.summary}\n\n"
                        content += f"Full Content:\n{page.content[:8000]}"
                        return content
                    except Exception:
                        continue
                except wikipedia.PageError:
                    continue

            return f"Could not retrieve a valid Wikipedia page for: {search_query}"

        except Exception as e:
            return f"Error fetching Wikipedia content: {str(e)}"