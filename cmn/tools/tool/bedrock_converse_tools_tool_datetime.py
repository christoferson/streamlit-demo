"""
cmn/tools/tool/bedrock_converse_tools_datetime.py
==================================================
Tool: datetime_tool

Provides date and time operations using the system clock.

Supported operations
--------------------
current_datetime      → current date and time  (YYYY-MM-DD HH:MM:SS)
current_date          → current date           (YYYY-MM-DD)
current_time          → current time           (HH:MM:SS)
current_timezone      → local timezone name
datetime_in_timezone  → date+time in a named tz (requires 'timezone' param)
                        e.g. America/New_York, Australia/Melbourne, Europe/London

Dependencies: pytz
"""

import logging
from datetime import datetime
import pytz

from cmn.tools.tool.bedrock_converse_tools_tool import AbstractBedrockConverseTool

logger = logging.getLogger(__name__)

_SUPPORTED_OPERATIONS = [
    "current_datetime",
    "current_date",
    "current_time",
    "current_timezone",
    "datetime_in_timezone",
]


class DateTimeBedrockConverseTool(AbstractBedrockConverseTool):

    def __init__(self):
        name = "datetime_tool"
        definition = {
            "toolSpec": {
                "name": name,
                "description": (
                    "Useful for getting date and time related information. "
                    "Supported operations: "
                    "current_datetime — current date and time; "
                    "current_date — current date; "
                    "current_time — current time; "
                    "current_timezone — local timezone name; "
                    "datetime_in_timezone — date and time in a specific timezone "
                    "(requires the 'timezone' parameter, "
                    "e.g. America/New_York, Australia/Melbourne, Europe/London)."
                ),
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "expression": {
                                "type": "string",
                                "enum": _SUPPORTED_OPERATIONS,
                                "description": (
                                    "The operation to perform. "
                                    "One of: current_datetime, current_date, "
                                    "current_time, current_timezone, "
                                    "datetime_in_timezone."
                                ),
                            },
                            "timezone": {
                                "type":        "string",
                                "description": (
                                    "Required for datetime_in_timezone. "
                                    "Example: America/New_York, "
                                    "Australia/Melbourne, Europe/London."
                                ),
                            },
                        },
                        "required": ["expression"],
                    }
                },
            }
        }
        super().__init__(name, definition)

    def summary(self) -> str:
        return "datetime_tool : get current date, time or timezone information"

    def invoke(self, params, tool_args=None):
        args      = tool_args or {}
        operation = args.get("expression", "").strip()

        if not operation:
            return "No operation provided. Pass 'expression' with one of: " \
                   + ", ".join(_SUPPORTED_OPERATIONS)

        logger.info("DateTimeTool: operation=%s", operation)

        try:
            if operation == "current_datetime":
                return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            elif operation == "current_date":
                return datetime.now().strftime("%Y-%m-%d")

            elif operation == "current_time":
                return datetime.now().strftime("%H:%M:%S")

            elif operation == "current_timezone":
                return datetime.now().astimezone().tzname()

            elif operation == "datetime_in_timezone":
                timezone = args.get("timezone", "").strip()
                if not timezone:
                    return (
                        "Error: 'timezone' parameter is required for "
                        "datetime_in_timezone. "
                        "Example: America/New_York"
                    )
                tz = pytz.timezone(timezone)
                return datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S %Z")

            else:
                return (
                    f"Unknown operation: '{operation}'. "
                    f"Supported: {', '.join(_SUPPORTED_OPERATIONS)}"
                )

        except pytz.exceptions.UnknownTimeZoneError as e:
            logger.warning("DateTimeTool: unknown timezone=%s", e)
            return f"Error: Unknown timezone '{e}'. Use a valid tz name e.g. America/New_York."

        except Exception as e:
            logger.error("DateTimeTool: unexpected error operation=%s error=%s", operation, e)
            return f"Error performing datetime operation: {str(e)}"