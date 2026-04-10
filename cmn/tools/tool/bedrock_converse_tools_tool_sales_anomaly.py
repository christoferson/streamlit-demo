"""
cmn/tools/tool/bedrock_converse_tools_sales_anomaly.py
=======================================================
Tool: sales_anomaly_detector

Detects anomalies in sales data using the Z-score method.
Flags months where a metric deviates beyond a configurable threshold.

Data source: _SALES_DF imported from bedrock_converse_tools_sales
Pure pandas/numpy — no streamlit, no external API calls.

invoke() returns a plain dict — the UI/renderer layer handles display.
"""

import logging

import numpy as np
import pandas as pd

from cmn.tools.tool.bedrock_converse_tools_tool import AbstractBedrockConverseTool
from cmn.tools.tool.bedrock_converse_tools_tool_sales import _SALES_DF

logger = logging.getLogger(__name__)


################################################################################
# SECTION: Helpers
################################################################################

def _severity(zscore: float, threshold: float) -> str:
    """
    Classify severity based on how far beyond the threshold the z-score is.

    Parameters
    ----------
    zscore    : the z-score of the data point
    threshold : the configured anomaly threshold

    Returns
    -------
    "mild" | "moderate" | "severe"
    """
    distance = abs(zscore) - threshold
    if distance < 0.5:
        return "mild"
    elif distance < 1.0:
        return "moderate"
    else:
        return "severe"


################################################################################
# SECTION: Tool
################################################################################

class SalesAnomalyBedrockConverseTool(AbstractBedrockConverseTool):
    """
    Detects anomalies in sales metrics for a given year.

    Method
    ------
    Z-score: flags months where abs(zscore) >= threshold.
    Default threshold = 1.5 (moderate sensitivity).

    Flow
    ----
    1. Filter _SALES_DF by year, region, category
    2. Aggregate metric by month
    3. Compute z-scores and pct_vs_mean
    4. Split into anomalies / normal lists
    5. Return payload — renderer layer handles display
    """

    def __init__(self):
        name = "sales_anomaly_detector"
        definition = {
            "toolSpec": {
                "name": name,
                "description": (
                    "Detects anomalies in sales metrics for a given year. "
                    "Flags months where performance is unusually high or low "
                    "compared to the yearly average. "
                    "Use this to identify root causes of underperformance."
                ),
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "year": {
                                "type":        "integer",
                                "description": "Year to analyse. Example: 2024",
                            },
                            "metric": {
                                "type": "string",
                                "enum": [
                                    "revenue",
                                    "units_sold",
                                    "returns",
                                    "gross_profit",
                                    "net_revenue",
                                ],
                                "description": "The metric to check for anomalies.",
                            },
                            "threshold": {
                                "type":        "number",
                                "description": (
                                    "Z-score threshold to flag as anomaly. "
                                    "1.0 = sensitive, 1.5 = moderate (default), 2.0 = strict."
                                ),
                            },
                            "region": {
                                "type":        "string",
                                "enum":        ["North", "South", "all"],
                                "description": "Filter by region. Defaults to all.",
                            },
                            "category": {
                                "type":        "string",
                                "enum":        ["Electronics", "Accessories", "all"],
                                "description": "Filter by category. Defaults to all.",
                            },
                        },
                        "required": ["year", "metric"],
                    }
                },
            }
        }
        super().__init__(name, definition)

    def summary(self) -> str:
        return (
            "sales_anomaly_detector : detects unusually high or low months "
            "in sales metrics using statistical analysis. "
            "Use this to identify root causes of underperformance or unexpected spikes."
        )

    def invoke(self, params, tool_args: dict = None) -> dict:
        args      = tool_args or {}
        year      = args.get("year")
        metric    = args.get("metric",    "revenue")
        threshold = args.get("threshold", 1.5)
        region    = args.get("region",    "all")
        category  = args.get("category",  "all")

        logger.info(
            "SalesAnomalyTool — year=%s metric=%s threshold=%s",
            year, metric, threshold,
        )

        return self._detect_anomalies(year, metric, threshold, region, category)

    # ── Private ───────────────────────────────────────────────────────────────

    def _detect_anomalies(
        self,
        year:      int,
        metric:    str,
        threshold: float,
        region:    str,
        category:  str,
    ) -> dict:
        """
        Core anomaly detection logic.

        Parameters
        ----------
        year      : year to filter on
        metric    : column to analyse
        threshold : z-score cutoff for flagging
        region    : region filter ("all" = no filter)
        category  : category filter ("all" = no filter)

        Returns
        -------
        dict with keys: year, metric, threshold, region, category,
                        mean, std, anomaly_count, anomalies, normal, summary
        """

        # ── Filter ────────────────────────────────────────────────────────────
        df = _SALES_DF[_SALES_DF["year"] == year].copy()

        if df.empty:
            return {
                "error":     f"No data found for year={year}",
                "anomalies": [],
            }

        if region != "all":
            df = df[df["region"] == region]
        if category != "all":
            df = df[df["category"] == category]

        if df.empty:
            return {
                "error":     f"No data after filters region={region} category={category}",
                "anomalies": [],
            }

        # ── Aggregate by month ────────────────────────────────────────────────
        monthly = (
            df.groupby(["month", "month_name"])[metric]
            .sum()
            .reset_index()
            .sort_values("month")
            .fillna(0)
        )

        # ── Z-score ───────────────────────────────────────────────────────────
        mean = monthly[metric].mean()
        std  = monthly[metric].std()

        if std == 0:
            return {
                "error":     "No variance in data — cannot compute anomalies",
                "anomalies": [],
            }

        monthly["zscore"]      = ((monthly[metric] - mean) / std).round(2)
        monthly["pct_vs_mean"] = (((monthly[metric] - mean) / mean) * 100).round(1)

        # ── Classify ──────────────────────────────────────────────────────────
        anomalies = []
        normal    = []

        for _, row in monthly.iterrows():
            entry = {
                "month":       int(row["month"]),
                "month_name":  row["month_name"],
                "value":       round(float(row[metric]), 2),
                "zscore":      float(row["zscore"]),
                "pct_vs_mean": float(row["pct_vs_mean"]),
            }

            if row["zscore"] <= -threshold:
                entry["flag"]     = "below_normal"
                entry["severity"] = _severity(row["zscore"], threshold)
                anomalies.append(entry)

            elif row["zscore"] >= threshold:
                entry["flag"]     = "above_normal"
                entry["severity"] = _severity(row["zscore"], threshold)
                anomalies.append(entry)

            else:
                entry["flag"] = "normal"
                normal.append(entry)

        # ── Sort worst first ──────────────────────────────────────────────────
        anomalies.sort(key=lambda x: abs(x["zscore"]), reverse=True)

        return {
            "year":          year,
            "metric":        metric,
            "threshold":     threshold,
            "region":        region,
            "category":      category,
            "mean":          round(float(mean), 2),
            "std":           round(float(std),  2),
            "anomaly_count": len(anomalies),
            "anomalies":     anomalies,
            "normal":        normal,
            "summary": {
                "worst_month": anomalies[0]["month_name"] if anomalies else "none",
                "best_month":  next(
                    (a["month_name"] for a in anomalies if a["flag"] == "above_normal"),
                    "none",
                ),
                "total_anomalies": len(anomalies),
            },
        }