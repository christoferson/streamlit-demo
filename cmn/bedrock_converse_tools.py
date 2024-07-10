import json

import ast
import operator as op


class AbstractBedrockConverseTool:
    definition:object
    def __init__(self, name, definition):
        self.name = name
        self.definition = definition

    def matches(self, name):
        return self.name == name
    
    def invoke(self, params):
        pass

class CalculatorBedrockConverseTool(AbstractBedrockConverseTool):

    # supported operators
    operators = {
        ast.Add: op.add, 
        ast.Sub: op.sub, 
        ast.Mult: op.mul,
        ast.Div: op.truediv, 
        ast.Pow: op.pow, 
        ast.BitXor: op.xor,
        ast.USub: op.neg
    }

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

    def invoke(self, params):
        return self.eval_(ast.parse(params, mode='eval').body)

    def eval_(self, node):
        if isinstance(node, ast.Constant):
            if isinstance(node.value, int):
                return node.value  # integer
            elif isinstance(node.value, float):
                return node.value  # integer
            else:
                return node.value  # integer
        elif isinstance(node, ast.BinOp):
            left = self.eval_(node.left)
            right = self.eval_(node.right)
            return self.operators[type(node.op)](left, right)
        elif isinstance(node, ast.UnaryOp):
            operand = self.eval_(node.operand)
            return self.operators[type(node.op)](operand)
        elif isinstance(node, ast.Call):
            func = self.eval_(node.func)
            args = [self.eval_(arg) for arg in node.args]
            keywords = {kw.arg: self.eval_(kw.value) for kw in node.keywords}
            return func(*args, **keywords)
        elif isinstance(node, ast.Name):
            return self.lookup_variable(node.id)
        else:
            raise TypeError(node)
    
    def lookup_variable(self, name):
        if name == 'round':
            return round
        raise NameError(f"Name {name} is not defined")



