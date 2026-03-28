import logging
from datetime import datetime
from cmn.tools.tool.bedrock_converse_tools_tool import AbstractBedrockConverseTool

logger = logging.getLogger(__name__)


class DateTimeBedrockConverseTool(AbstractBedrockConverseTool):

    def __init__(self):
        name = "datetime"
        definition = {
            "toolSpec": {
                "name": name,
                "description": (
                    "Returns the current date and time. "
                    "Use this to answer any question about today's date, "
                    "current time, day of week, or whether today is a holiday."
                ),
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "timezone": {
                                "type": "string",
                                "description": (
                                    "Optional timezone name. "
                                    "Example: 'US/Eastern', 'UTC'. "
                                    "Defaults to local system time."
                                ),
                            }
                        },
                        "required": [],
                    }
                },
            }
        }
        super().__init__(name, definition)

    def summary(self) -> str:
        return "datetime : use to get the current date, time and day of week"

    def invoke(self, params, tool_args: dict = None) -> dict:
        now = datetime.now()
        logger.info("DateTimeTool invoked: %s", now)
        return {
            "date":        now.strftime("%Y-%m-%d"),
            "time":        now.strftime("%H:%M:%S"),
            "day_of_week": now.strftime("%A"),
            "month":       now.strftime("%B"),
            "year":        now.year,
            "iso":         now.isoformat(),
        }