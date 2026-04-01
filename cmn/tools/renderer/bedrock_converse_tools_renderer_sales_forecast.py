import pandas as pd
import plotly.express as px
import streamlit as st
from typing import Any
from cmn.tools.renderer.bedrock_converse_tools_renderer import AbstractToolRenderer


class SalesForecastToolRenderer(AbstractToolRenderer):

    @property
    def tool_name(self) -> str:
        return "sales_forecast"

    def render(self, tool_args: dict, tool_result: Any, result_container) -> None:

        if "error" in tool_result:
            with result_container:
                st.warning(f"Forecast error: {tool_result['error']}")
            return

        actuals       = tool_result.get("actuals",  [])
        forecast      = tool_result.get("forecast", [])
        metric        = tool_result.get("metric",   "revenue")
        train_year    = tool_result.get("train_year")
        forecast_year = tool_result.get("forecast_year")
        model_info    = tool_result.get("model",    {})
        summary       = tool_result.get("summary",  {})

        with result_container:
            st.markdown(f"**📈 Sales Forecast — {metric.title()}**")

            col1, col2, col3 = st.columns(3)
            col1.metric("Trend",      summary.get("trend", "").title())
            col2.metric("Monthly Δ",  f"{model_info.get('slope', 0):+,.0f}")
            col3.metric("Confidence", summary.get("confidence", "").title(),
                        delta=f"R²={model_info.get('r2_score', 0)}")

            rows = []
            for row in actuals:
                rows.append({
                    "order":  row["month"],
                    "label":  f"{row['month_name']} {train_year}",
                    "value":  row["actual"],
                    "series": f"{train_year} Actual",
                })
            for i, row in enumerate(forecast):
                rows.append({
                    "order":  12 + (i + 1),
                    "label":  f"{row['month_name']} {forecast_year}",
                    "value":  row["forecasted_value"],
                    "series": f"{forecast_year} Forecast",
                })

            plot_df = pd.DataFrame(rows).sort_values("order")

            st.markdown(f"**{train_year} Actuals vs {forecast_year} Forecast**")
            fig = px.line(
                plot_df, x="label", y="value", color="series",
                markers=True,
                color_discrete_map={
                    f"{train_year} Actual":      "#4A90D9",
                    f"{forecast_year} Forecast": "#FF6B6B",
                },
            )
            fig.update_xaxes(
                categoryorder="array",
                categoryarray=plot_df["label"].tolist(),
            )
            st.plotly_chart(fig, width="content")

            st.markdown(f"**{forecast_year} Monthly Forecast**")
            st.dataframe(
                pd.DataFrame([
                    {
                        "Month":                f"{f['month_name']} {forecast_year}",
                        f"Forecast ({metric})": f"{f['forecasted_value']:,.0f}",
                    }
                    for f in forecast
                ]),
                width="content",
                hide_index=True,
            )