import pandas as pd
import logging
from cmn.tools.tool.bedrock_converse_tools_tool import AbstractBedrockConverseTool

logger = logging.getLogger(__name__)


class EDAProfileBedrockConverseTool(AbstractBedrockConverseTool):

    def __init__(self):
        name = "eda_profile"
        definition = {
            "toolSpec": {
                "name": name,
                "description": (
                    "Profiles a dataset — shape, column types, null counts, "
                    "basic statistics for numeric columns, "
                    "and top values for categorical columns. "
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
                                    "Data to profile. "
                                    "Pass the data array from another tool result. "
                                    "Example: pass data array from sales_data result."
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
            "eda_profile : profiles a dataset — shape, nulls, "
            "basic stats for numeric columns, top values for categorical. "
            "Pass data array from another tool result."
        )

    def invoke(self, params, tool_args: dict = None) -> dict:
        args = tool_args or {}
        data = args.get("data", [])

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

        logger.info("EDAProfileTool: shape=%s columns=%s", df.shape, list(df.columns))

        num_df = df.select_dtypes(include="number")
        cat_df = df.select_dtypes(include=["object", "category"])

        numeric = []
        for col in num_df.columns:
            s = num_df[col].dropna()
            if len(s) == 0:
                continue
            numeric.append({
                "column": col,
                "count":  int(s.count()),
                "nulls":  int(df[col].isna().sum()),
                "mean":   round(float(s.mean()), 2),
                "median": round(float(s.median()), 2),
                "std":    round(float(s.std()), 2),
                "min":    round(float(s.min()), 2),
                "max":    round(float(s.max()), 2),
            })

        categorical = []
        for col in cat_df.columns:
            s  = cat_df[col].dropna()
            vc = s.value_counts()
            categorical.append({
                "column":    col,
                "unique":    int(s.nunique()),
                "nulls":     int(df[col].isna().sum()),
                "top_value": str(vc.index[0]) if len(vc) > 0 else "N/A",
                "top_3":     {str(k): int(v) for k, v in vc.head(3).items()},
            })

        return {
            "rows":           len(df),
            "columns":        len(df.columns),
            "column_names":   list(df.columns),
            "total_nulls":    int(df.isna().sum().sum()),
            "duplicate_rows": int(df.duplicated().sum()),
            "numeric":        numeric,
            "categorical":    categorical,
        }