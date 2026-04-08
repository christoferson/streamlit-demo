"""
cmn/tools/tool/bedrock_converse_tools_tool_sales_kpi.py
===================================================
Tool: render_sales_kpi

Renders KPI metric cards on the dashboard.
Does NOT compute anything — the LLM computes deltas from sales_data results
and passes pre-computed values here for display only.

invoke() returns a plain dict — the UI/renderer layer handles display.
No streamlit imports.
"""

import logging
from cmn.tools.tool.bedrock_converse_tools_tool import AbstractBedrockConverseTool

logger = logging.getLogger(__name__)


class SalesKpiBedrockConverseTool(AbstractBedrockConverseTool):
    """
    Returns a kpi_ready payload for the renderer layer.
    Supports up to 4 metrics per row.
    """

    def __init__(self):
        name = "render_sales_kpi"
        definition = {
            "toolSpec": {
                "name": name,
                "description": (
                    "Renders KPI metric cards on the dashboard. "
                    "Call this after fetching sales data to display key metrics visually. "
                    "You must compute all values and deltas yourself before calling this tool. "
                    "Supports up to 4 metrics per row."
                ),
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "title": {
                                "type":        "string",
                                "description": "Section heading above the KPI cards. Example: 2024 vs 2023 Sales Summary",
                            },
                            "metrics": {
                                "type":        "array",
                                "description": "List of KPI metrics to display as cards.",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "label": {
                                            "type":        "string",
                                            "description": "Metric name. Example: Total Revenue",
                                        },
                                        "value": {
                                            "type":        "string",
                                            "description": "Formatted metric value. Example: $3.13M or 14,905",
                                        },
                                        "delta": {
                                            "type":        "string",
                                            "description": (
                                                "Formatted change vs comparison period. "
                                                "Example: -$97K vs 2023 or +5.2%"
                                            ),
                                        },
                                        "delta_color": {
                                            "type":        "string",
                                            "enum":        ["normal", "inverse", "off"],
                                            "description": (
                                                "normal  → green=positive, red=negative. Use for revenue, profit, units. "
                                                "inverse → red=positive, green=negative. Use for returns, costs, complaints. "
                                                "off     → always grey. Use for neutral metrics like avg order size."
                                            ),
                                        },
                                    },
                                    "required": ["label", "value"],
                                },
                            },
                        },
                        "required": ["metrics"],
                    }
                },
            }
        }
        super().__init__(name, definition)

    def summary(self) -> str:
        return (
            "render_sales_kpi : renders KPI metric cards on the dashboard. "
            "Call this after fetching sales data to display key metrics visually. "
            "Compute all values and deltas yourself before calling this tool."
        )

    def invoke(self, params, tool_args: dict = None) -> dict:
        args    = tool_args or {}
        metrics = args.get("metrics", [])
        title   = args.get("title",   "")

        if not metrics:
            return {"error": "No metrics provided"}

        logger.info("SalesKpiTool — title=%s metrics=%d", title, len(metrics))

        # Return payload — renderer layer handles display
        return {
            "status":  "kpi_ready",
            "title":   title,
            "metrics": metrics,
        }