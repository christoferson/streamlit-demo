import ast
import operator as op
import logging
from cmn.tools.tool.bedrock_converse_tools_tool import AbstractBedrockConverseTool

logger = logging.getLogger(__name__)


class CalculatorBedrockConverseTool(AbstractBedrockConverseTool):

    operators = {
        ast.Add:  op.add,
        ast.Sub:  op.sub,
        ast.Mult: op.mul,
        ast.Div:  op.truediv,
        ast.Pow:  op.pow,
        ast.BitXor: op.xor,
        ast.USub: op.neg,
    }

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
            result = self.eval_(ast.parse(expression, mode='eval').body)
            return {"result": result}
        except Exception as e:
            return {"error": str(e)}

    def eval_(self, node):
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.BinOp):
            return self.operators[type(node.op)](
                self.eval_(node.left),
                self.eval_(node.right),
            )
        elif isinstance(node, ast.UnaryOp):
            return self.operators[type(node.op)](self.eval_(node.operand))
        elif isinstance(node, ast.Call):
            func     = self.eval_(node.func)
            args     = [self.eval_(a) for a in node.args]
            keywords = {kw.arg: self.eval_(kw.value) for kw in node.keywords}
            return func(*args, **keywords)
        elif isinstance(node, ast.Name):
            return self._lookup(node.id)
        else:
            raise TypeError(node)

    def _lookup(self, name: str):
        if name == "round":
            return round
        raise NameError(f"Name '{name}' is not defined")