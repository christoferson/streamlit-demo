import logging
from cmn.bedrock_converse_tools import AbstractBedrockConverseTool

logger = logging.getLogger(__name__)


class ChartBedrockConverseTool(AbstractBedrockConverseTool):
    """
    Does NOT render the chart itself.
    Returns a structured chart payload that the UI layer renders.
    """

    def __init__(self):
        name = "render_chart"
        definition = {
            "toolSpec": {
                "name": name,
                "description": (
                    "Use this tool to visualize data as a chart. "
                    "Call this AFTER you have fetched the data. "
                    "Pass the data you want to chart along with chart configuration."
                ),
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "chart_type": {
                                "type": "string",
                                "enum": ["bar", "line", "area"],
                                "description": "Type of chart to render.",
                            },
                            "title": {
                                "type": "string",
                                "description": "Chart title. Example: 2024 Monthly Revenue",
                            },
                            "x_label": {
                                "type": "string",
                                "description": "Column name to use as X axis. Example: month_name",
                            },
                            "y_label": {
                                "type": "string",
                                "description": "Column name to use as Y axis. Example: revenue",
                            },
                            "data": {
                                "type": "array",
                                "description": (
                                    "Array of data objects to chart. "
                                    "Example: [{'month_name': 'January', 'revenue': 125000}, ...]"
                                ),
                                "items": {"type": "object"},
                            },
                            "color_series": {                          # ← replaces "color"
                                "type": "string",
                                "description": (
                                    "Column name to use for color grouping and legend. "
                                    "Required when data contains multiple series. "
                                    "The column must exist in the data array. "
                                    "Example: if data has a 'year' column with values "
                                    "'2023' and '2024', set color_series='year' to get "
                                    "one colored line/bar per year with a legend."
                                ),
                            },
                        },
                        "required": ["chart_type", "title", "x_label", "y_label", "data"],
                    }
                },
            }
        }
        super().__init__(name, definition)

    def summary(self) -> str:
        return (
            "render_chart : renders a visual chart or graph. "
            "Use when user asks for a chart, plot, bar chart, line chart or visualization. "
            "x_label and y_label must exactly match column names from the data. "
            "Set color_series to the column name that identifies different series "
            "(e.g. 'year', 'region', 'category') to get separate colored lines and a legend. "
            "When charting forecast results, always include historical actuals AND "
            "forecast values together in the data array — never forecast only."
        )

    def invoke(self, params, tool_args: dict = None) -> dict:
        """
        Just validates and returns the payload.
        Actual rendering happens in the UI layer.
        """
        args = tool_args or {}

        if not args.get("data"):
            return {"error": "No data provided to chart"}

        logger.info(
            "ChartTool — type=%s title=%s rows=%d color_series=%s",
            args.get("chart_type"),
            args.get("title"),
            len(args.get("data", [])),
            args.get("color_series"),
        )

        return {
            "status":       "chart_ready",
            "chart_type":   args.get("chart_type"),
            "title":        args.get("title"),
            "x_label":      args.get("x_label"),
            "y_label":      args.get("y_label"),
            "data":         args.get("data"),
            "color_series": args.get("color_series"),   # ← replaces "color"
        }