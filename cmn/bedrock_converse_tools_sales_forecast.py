import pandas as pd
import numpy as np
import logging
from sklearn.linear_model import LinearRegression
from cmn.bedrock_converse_tools import AbstractBedrockConverseTool

logger = logging.getLogger(__name__)

MONTH_NAMES = {
    1: "January",  2: "February", 3: "March",     4: "April",
    5: "May",      6: "June",     7: "July",       8: "August",
    9: "September",10: "October", 11: "November",  12: "December",
}


class SalesForecastBedrockConverseTool(AbstractBedrockConverseTool):
    """
    Forecasts future sales using linear regression.
    Receives historical data directly from the LLM — no DB import needed.
    LLM is responsible for fetching data first via sales_data tool,
    then passing it here.
    """

    def __init__(self):
        name = "sales_forecast"
        definition = {
            "toolSpec": {
                "name": name,
                "description": (
                    "Forecasts future monthly sales using linear regression. "
                    "You must provide the historical monthly data to train on — "
                    "fetch it first using sales_data tool, then pass the result here. "
                    "Returns forecasted values for the requested number of future months."
                ),
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "data": {
                                "type": "array",
                                "description": (
                                    "Historical monthly data to train on. "
                                    "Each item must have 'month' (1-12) and the metric field. "
                                    "Pass the data array directly from sales_data tool result."
                                ),
                                "items": {"type": "object"},
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
                                "description": "The metric column to forecast.",
                            },
                            "periods": {
                                "type": "integer",
                                "description": (
                                    "Number of future months to forecast. "
                                    "Example: 2 forecasts the next 2 months. Max 12."
                                ),
                            },
                            "train_year": {
                                "type": "integer",
                                "description": (
                                    "The year the training data belongs to. "
                                    "Used for labelling forecast output only."
                                ),
                            },
                        },
                        "required": ["data", "metric", "periods", "train_year"],
                    }
                },
            }
        }
        super().__init__(name, definition)

    def summary(self) -> str:
        return (
            "sales_forecast : forecasts future monthly sales using linear regression. "
            "Requires historical data passed directly — call sales_data first, "
            "then pass its data array here. "
            "The result contains both 'actuals' and 'forecast' arrays. "
            "When rendering a chart, combine BOTH actuals and forecast into one data array "
            "before calling render_chart — do not pass forecast data only."
        )

    def invoke(self, params, tool_args: dict = None) -> dict:
        args       = tool_args or {}
        data       = args.get("data", [])
        metric     = args.get("metric",     "revenue")
        periods    = min(int(args.get("periods") or 3), 12)
        train_year = args.get("train_year")

        logger.info(
            "SalesForecastTool — train_year=%s metric=%s periods=%s rows=%d",
            train_year, metric, periods, len(data),
        )

        if not data:
            return {"error": "No data provided. Call sales_data first and pass its data array."}

        return self._forecast(data, metric, periods, train_year)

    # ── Core Logic ────────────────────────────────────────────────────────────

    def _forecast(
        self,
        data:       list,
        metric:     str,
        periods:    int,
        train_year: int,
    ) -> dict:

        df = pd.DataFrame(data)

        # ── Validate ──────────────────────────────────────────────────────
        if "month" not in df.columns:
            return {"error": f"'month' column not found in data. Got: {list(df.columns)}"}

        if metric not in df.columns:
            return {"error": f"'{metric}' column not found in data. Got: {list(df.columns)}"}

        df = df[["month", metric]].dropna().sort_values("month")

        if len(df) < 3:
            return {"error": "Need at least 3 months of data to forecast"}

        # ── Train ─────────────────────────────────────────────────────────
        X_train = df["month"].values.reshape(-1, 1)
        y_train = df[metric].values.astype(float)

        model = LinearRegression()
        model.fit(X_train, y_train)

        r2    = round(model.score(X_train, y_train), 3)
        slope = model.coef_[0]

        # ── Predict ───────────────────────────────────────────────────────
        last_month    = int(df["month"].max())
        future_months = list(range(last_month + 1, last_month + periods + 1))
        predictions   = model.predict(np.array(future_months).reshape(-1, 1))

        forecast_year = (train_year or 0) + 1

        forecast_data = []
        for future_m, pred in zip(future_months, predictions):
            calendar_month = ((future_m - 1) % 12) + 1
            forecast_data.append({
                "month":             calendar_month,
                "month_name":        MONTH_NAMES[calendar_month],
                "forecast_year":     forecast_year,
                "forecasted_value":  round(max(float(pred), 0), 2),
            })

        # ── Actuals for reference ─────────────────────────────────────────
        actuals = [
            {
                "month":      int(row["month"]),
                "month_name": MONTH_NAMES[int(row["month"])],
                "actual":     round(float(row[metric]), 2),
            }
            for _, row in df.iterrows()
        ]

        trend = "upward" if slope > 0 else ("downward" if slope < 0 else "flat")

        return {
            "train_year":    train_year,
            "forecast_year": forecast_year,
            "metric":        metric,
            "periods":       periods,
            "model": {
                "type":       "linear_regression",
                "r2_score":   r2,
                "slope":      round(float(slope), 2),
                "intercept":  round(float(model.intercept_), 2),
                "trend":      trend,
            },
            "actuals":  actuals,
            "forecast": forecast_data,
            "summary": {
                "trend":          trend,
                "monthly_change": round(float(slope), 2),
                "avg_forecast":   round(float(np.mean([f["forecasted_value"] for f in forecast_data])), 2),
                "r2_score":       r2,
                "confidence":     _confidence_label(r2),
            },
        }


def _confidence_label(r2: float) -> str:
    if r2 >= 0.9:  return "high"
    elif r2 >= 0.7: return "moderate"
    elif r2 >= 0.5: return "low"
    else:           return "very_low — treat forecast with caution"