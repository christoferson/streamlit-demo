from cmn.bedrock_converse_tools import AbstractBedrockConverseTool
from datetime import datetime
import pytz

@DeprecationWarning
class DateTimeBedrockConverseTool(AbstractBedrockConverseTool):

    def __init__(self):
        name = "datetime_tool"
        definition = {
            "toolSpec": {
                "name": name,
                "description": """Useful for getting date and time related information.
                Supported operations:
                - current_datetime: Returns the current date and time
                - current_date: Returns the current date
                - current_time: Returns the current time
                - current_timezone: Returns the current timezone
                - datetime_in_timezone: Returns the current date and time in a specific timezone. Requires timezone parameter e.g. America/New_York, Australia/Melbourne, Europe/London
                """,
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "expression": {
                                "type": "string",
                                "description": "The operation to perform. One of: current_datetime, current_date, current_time, current_timezone, datetime_in_timezone"
                            },
                            "timezone": {
                                "type": "string",
                                "description": "The timezone to use for datetime_in_timezone operation. Example: America/New_York, Australia/Melbourne, Europe/London"
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
        return "datetime : use to get the current date and time"

    def invoke(self, expression, tool_args=None):
        try:
            operation = expression

            if operation == "current_datetime":
                return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            elif operation == "current_date":
                return datetime.now().strftime("%Y-%m-%d")

            elif operation == "current_time":
                return datetime.now().strftime("%H:%M:%S")

            elif operation == "current_timezone":
                return datetime.now().astimezone().tzname()

            elif operation == "datetime_in_timezone":
                timezone = tool_args.get('timezone') if tool_args else None
                if not timezone:
                    return "Error: timezone parameter is required for datetime_in_timezone operation"
                tz = pytz.timezone(timezone)
                return datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S %Z")

            else:
                return f"Unknown operation: {operation}"

        except pytz.exceptions.UnknownTimeZoneError as e:
            return f"Error: Unknown timezone: {str(e)}"
        except Exception as e:
            return f"Error performing datetime operation: {str(e)}"