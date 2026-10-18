import logging
from cmn.tools.tool.bedrock_converse_tools_tool import AbstractBedrockConverseTool
from cmn.tools.tool.tool_safety import evaluate_expression

logger = logging.getLogger(__name__)


class CalculatorBedrockConverseTool(AbstractBedrockConverseTool):

    def __init__(self):
        name = "expr_evaluator"
        definition = {
            "toolSpec": {
                "name": name,
                "description": (
                    "Useful for when you need to answer questions about math. "
                    "This tool is only for math questions and nothing else. "
                    "Only input math expressions."
                ),
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "expression": {
                                "type": "string",
                                "description": "Numerical expression. Example: 47.5 + 98.3",
                            }
                        },
                        "required": ["expression"],
                    }
                },
            }
        }
        super().__init__(name, definition)

    def summary(self) -> str:
        return "expr_evaluator : use for math and numerical calculations"

    def invoke(self, params, tool_args: dict = None) -> dict:
        print(f"Params: {params}, Tool Args: {tool_args}")
        expression = (tool_args or {}).get("expression")

        if not expression:
            return {"error": "No expression provided"}

        logger.info("CalculatorTool: expression=%s", expression)

        try:
            # Size-limited so e.g. 9**9**9 can't hang the app; see tool_safety.py
            return {"result": evaluate_expression(expression)}
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}"}