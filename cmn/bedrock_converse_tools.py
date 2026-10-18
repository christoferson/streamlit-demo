import json

from cmn.tools.tool.tool_safety import evaluate_expression

# Deprecated: superseded by cmn/tools/tool/
class AbstractBedrockConverseTool:
    definition:object
    def __init__(self, name, definition):
        self.name = name
        self.definition = definition

    def matches(self, name):
        return self.name == name
    
    def invoke(self, params, tool_args=None):
        raise NotImplementedError
    
    def summary(self) -> str | None:
        """
        Override in subclass to provide a one-line description
        of when and how to use this tool.
        Returns None by default — excluded from system prompt if not overridden.
        """
        return None

class CalculatorBedrockConverseTool(AbstractBedrockConverseTool):

    def __init__(self):
        name = "expr_evaluator"
        definition = {
                "toolSpec": {
                    "name": name,
                    "description": """Useful for when you need to answer questions about math. This tool is only for math questions and nothing else. Only input
            math expressions.""",
                    "inputSchema": {
                        "json": {
                            "type": "object",
                            "properties": {
                                "expression": {
                                    "type": "string",
                                    "description": "Numerical Expresion. Example 47.5 + 98.3."
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
    
    def summary(self) -> str:
        return "expr_evaluator : use for math and numerical calculations"

    def invoke(self, params, tool_args=None):
        expression = tool_args.get("expression") if tool_args else None
        if not expression:
            return {"error": "No expression provided"}
        try:
            return evaluate_expression(expression)  # size-limited, see cmn/tools/tool/tool_safety.py
        except Exception as err:  # syntax errors, unsupported operators, limits, division by zero, ...
            return {"error": f"{type(err).__name__}: {err}"}
