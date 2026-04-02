import streamlit as st
from typing import Any
from cmn.tools.renderer.bedrock_converse_tools_renderer import AbstractToolRenderer


class SalesAnomalyToolRenderer(AbstractToolRenderer):

    @property
    def tool_name(self) -> str:
        return "sales_anomaly_detector"

    def render(self, tool_args: dict, tool_result: Any, result_container) -> None:

        if "error" in tool_result:
            with result_container:
                st.warning(f"Anomaly detection: {tool_result['error']}")
            return

        anomalies = tool_result.get("anomalies", [])
        normal    = tool_result.get("normal",    [])
        metric    = tool_result.get("metric",    "revenue")
        year      = tool_result.get("year",      "")
        mean      = tool_result.get("mean",       0)

        with result_container:
            st.markdown(f"**🔍 Anomaly Detection — {metric.title()} {year}**")
            st.caption(
                f"Yearly mean: {mean:,.0f} | "
                f"Threshold: ±{tool_result.get('threshold', 1.5)} std dev"
            )

            if not anomalies:
                st.success("✅ No anomalies detected — all months within normal range.")
                return

            cols = st.columns(min(len(anomalies), 4))
            for i, entry in enumerate(anomalies):
                with cols[i % 4]:
                    icon = "🔴" if entry["flag"] == "below_normal" else "🟢"
                    st.metric(
                        label       = f"{icon} {entry['month_name']}",
                        value       = f"{entry['value']:,.0f}",
                        delta       = f"{entry['pct_vs_mean']:+.1f}% vs mean",
                        delta_color = "normal" if entry["flag"] == "above_normal" else "inverse",
                    )
                    st.caption(f"Z-score: {entry['zscore']} | {entry.get('severity', '')}")

            if normal:
                st.caption(f"✅ Normal months: {', '.join(n['month_name'] for n in normal)}")