from cmn.bedrock_converse_tools import AbstractBedrockConverseTool

@DeprecationWarning
class AcronymBedrockConverseTool(AbstractBedrockConverseTool):

 
    def __init__(self):
        name = "acronym_evaluator"
        definition = {
                "toolSpec": {
                    "name": name,
                    "description": """Useful for when you need to decode proprietary acronym. 
                    This tool is only resolving meanings of acronyms that are not generally defined.
                    Tool will return the work 'unknown' is it does not know the answer.""",
                    "inputSchema": {
                        "json": {
                            "type": "object",
                            "properties": {
                                "expression": {
                                    "type": "string",
                                    "description": "Acronym. Example AUP,POIE."
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
        return "acronym : use to look up acronym definitions"

    def invoke(self, params, tool_args=None):
        if params == "AUP":
            return "Access Utilization Procedure"
        elif params == "POIE":
            return "Procedural Oversight and Inspection Evaluation"
        elif params == "SPT":
            return "Sustainable Production Technologies"
        return "unknown"

