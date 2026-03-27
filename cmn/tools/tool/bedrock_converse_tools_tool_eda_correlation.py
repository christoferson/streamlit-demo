import pandas as pd
import logging
from cmn.tools.tool.bedrock_converse_tools_tool import AbstractBedrockConverseTool

logger = logging.getLogger(__name__)


class EDACorrelationBedrockConverseTool(AbstractBedrockConverseTool):

    def __init__(self):
        name = "eda_correlation"
        definition = {
            "toolSpec": {
                "name": name,
                "description": (
                    "Calculates correlation between numeric columns in a dataset. "
                    "Returns notable correlations (|r| >= 0.5) ranked by strength. "
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
                            "columns": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": (
                                    "Optional. Specific columns to correlate. "
                                    "Uses all numeric columns if not specified."
                                ),
                            },
                        },
                        "required": ["data"],
                    }
                },
            }
        }
        super().__init__(name, definition)

    def summary(self) -> str:
        return (
            "eda_correlation : finds correlations between numeric columns. "
            "Returns notable correlations ranked by strength. "
            "Pass data array from another tool result."
        )

    def invoke(self, params, tool_args: dict = None) -> dict:
        args    = tool_args or {}
        data    = args.get("data", [])
        columns = args.get("columns", [])

        if not data:
            return {
                "error": (
                    "No data provided. "
                    "Pass data array from another tool result. "
                    "Example: call sales_data first then pass its data array here."
                )
            }

        df = pd.DataFrame(data)

        if df.empty:
            return {"error": "Data is empty."}

        logger.info("EDACorrelationTool: shape=%s", df.shape)

        # ── Select columns ────────────────────────────────────────────────
        num_df = (
            df[columns].select_dtypes(include="number")
            if columns
            else df.select_dtypes(include="number")
        )

        if num_df.shape[1] < 2:
            return {
                "error": (
                    "Need at least 2 numeric columns for correlation. "
                    f"Available numeric: {list(df.select_dtypes(include='number').columns)}"
                )
            }

        # ── Drop constant columns ─────────────────────────────────────────
        num_df = num_df.loc[:, num_df.std() > 0]

        if num_df.shape[1] < 2:
            return {"error": "Need at least 2 non-constant numeric columns."}

        # ── Compute correlation ───────────────────────────────────────────
        corr = num_df.corr()
        cols = list(corr.columns)

        notable = []
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                val = corr.iloc[i, j]
                if abs(val) >= 0.5:
                    notable.append({
                        "col1":        cols[i],
                        "col2":        cols[j],
                        "correlation": round(float(val), 4),
                        "strength": (
                            "strong positive"   if val >= 0.7  else
                            "moderate positive" if val >= 0.5  else
                            "strong negative"   if val <= -0.7 else
                            "moderate negative"
                        ),
                    })

        notable.sort(key=lambda x: abs(x["correlation"]), reverse=True)

        return {
            "columns_analyzed":     cols,
            "notable_correlations": notable[:15],
            "total_notable":        len(notable),
            "summary": (
                f"Analyzed {len(cols)} numeric columns. "
                f"Found {len(notable)} notable correlations (|r| >= 0.5)."
            ),
        }