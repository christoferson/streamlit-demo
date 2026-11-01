"""
cmn/tools/tool/bedrock_converse_tools_acronym.py
=================================================
Tool: acronym_evaluator

Resolves proprietary/internal acronyms that are not generally known.
Returns 'unknown' if the acronym is not in the lookup table.

To extend: add entries to _ACRONYM_TABLE.
"""

import logging
from cmn.tools.tool.bedrock_converse_tools_tool import AbstractBedrockConverseTool

logger = logging.getLogger(__name__)

_ACRONYM_TABLE = {
    "AUP":  "Access Utilization Procedure",
    "POIE": "Procedural Oversight and Inspection Evaluation",
    "SPT":  "Sustainable Production Technologies",
}


class AcronymBedrockConverseTool(AbstractBedrockConverseTool):

    def __init__(self):
        name = "acronym_evaluator"
        definition = {
            "toolSpec": {
                "name": name,
                "description": (
                    "Useful for when you need to decode proprietary acronyms. "
                    "This tool only resolves meanings of acronyms that are not "
                    "generally defined. "
                    "Returns 'unknown' if the acronym is not recognised."
                ),
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "expression": {
                                "type":        "string",
                                "description": "Acronym to look up. Example: AUP, POIE",
                            }
                        },
                        "required": ["expression"],
                    }
                },
            }
        }
        super().__init__(name, definition)

    def summary(self) -> str:
        return "acronym_evaluator : look up proprietary acronym definitions"

    def invoke(self, params, tool_args=None):
        args       = tool_args or {}
        expression = args.get("expression", "").strip().upper()

        if not expression:
            return "No acronym provided. Pass 'expression' with an acronym to look up."

        logger.info("AcronymTool: expression=%s", expression)

        result = _ACRONYM_TABLE.get(expression, "unknown")
        logger.info("AcronymTool: result=%s", result)
        return result