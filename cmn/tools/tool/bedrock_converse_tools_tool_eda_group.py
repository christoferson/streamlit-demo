import pandas as pd
import logging
from scipy import stats as scipy_stats
from cmn.tools.tool.bedrock_converse_tools_tool import AbstractBedrockConverseTool

logger = logging.getLogger(__name__)


class EDAGroupBedrockConverseTool(AbstractBedrockConverseTool):

    def __init__(self):
        name = "eda_group"
        definition = {
            "toolSpec": {
                "name": name,
                "description": (
                    "Compares a numeric metric across groups in a dataset. "
                    "Returns mean, median, min, max per group "
                    "and ANOVA test to check if differences are significant. "
                    "Pass data directly from another tool result such as sales_data."
                ),
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "data": {
                                "type": "array",
                                "items": {"type": "object"},
                                "description": (
                                    "Data to analyze. "
                                    "Pass the data array from another tool result. "
                                    "Example: pass data array from sales_data result."
                                ),
                            },
                            "group_by": {
                                "type": "string",
                                "description": (
                                    "Column to group by. "
                                    "Example: 'region', 'category', 'month_name'"
                                ),
                            },
                            "target": {
                                "type": "string",
                                "description": (
                                    "Numeric column to compare across groups. "
                                    "Example: 'revenue', 'units_sold', 'returns'"
                                ),
                            },
                        },
                        "required": ["data", "group_by", "target"],
                    }
                },
            }
        }
        super().__init__(name, definition)

    def summary(self) -> str:
        return (
            "eda_group : compares a numeric metric across groups. "
            "Returns stats per group and ANOVA significance test. "
            "Pass data array from another tool result."
        )

    def invoke(self, params, tool_args: dict = None) -> dict:
        args     = tool_args or {}
        data     = args.get("data", [])
        group_by = args.get("group_by", "")
        target   = args.get("target", "")

        if not data:
            return {
                "error": (
                    "No data provided. "
                    "Pass data array from another tool result. "
                    "Example: call sales_data first then pass its data array here."
                )
            }

        if not group_by:
            return {"error": "group_by is required."}

        if not target:
            return {"error": "target is required."}

        df = pd.DataFrame(data)

        if df.empty:
            return {"error": "Data is empty."}

        logger.info("EDAGroupTool: group_by=%s target=%s", group_by, target)

        if group_by not in df.columns:
            return {
                "error": f"group_by='{group_by}' not found. "
                         f"Available: {list(df.columns)}"
            }

        if target not in df.columns:
            return {
                "error": f"target='{target}' not found. "
                         f"Available: {list(df.columns)}"
            }

        # ── Group stats ───────────────────────────────────────────────────
        groups  = df.groupby(group_by)[target]
        summary = (
            groups
            .agg(count="count", mean="mean", median="median",
                 std="std", min="min", max="max")
            .round(2)
            .reset_index()
            .to_dict(orient="records")
        )

        # ── ANOVA ─────────────────────────────────────────────────────────
        anova        = None
        group_arrays = [
            g.dropna().values
            for _, g in groups
            if len(g.dropna()) > 1
        ]

        if len(group_arrays) >= 2:
            try:
                f, p   = scipy_stats.f_oneway(*group_arrays)
                anova  = {
                    "f_statistic": round(float(f), 4),
                    "p_value":     round(float(p), 6),
                    "significant": bool(p < 0.05),
                    "interpretation": (
                        f"Significant difference between groups (p={p:.4f})"
                        if p < 0.05
                        else f"No significant difference between groups (p={p:.4f})"
                    ),
                }
            except Exception as e:
                logger.warning("ANOVA failed: %s", e)

        # ── Best / worst ──────────────────────────────────────────────────
        means      = {r[group_by]: r["mean"] for r in summary}
        best_group = max(means, key=means.get)
        worst_group = min(means, key=means.get)

        return {
            "group_by":    group_by,
            "target":      target,
            "groups":      summary,
            "anova":       anova,
            "best_group":  best_group,
            "worst_group": worst_group,
            "summary": (
                f"Compared '{target}' across {len(summary)} groups of '{group_by}'. "
                f"Best: {best_group} ({round(means[best_group], 2)}), "
                f"Worst: {worst_group} ({round(means[worst_group], 2)})."
            ),
        }