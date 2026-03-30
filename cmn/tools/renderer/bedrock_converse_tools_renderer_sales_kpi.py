import streamlit as st
from typing import Any
from cmn.tools.renderer.bedrock_converse_tools_renderer import AbstractToolRenderer


class SalesKpiToolRenderer(AbstractToolRenderer):

    @property
    def tool_name(self) -> str:
        return "render_sales_kpi"

    def render(self, tool_args: dict, tool_result: Any, result_container) -> None:

        if tool_result.get("status") != "kpi_ready":
            return

        title   = tool_result.get("title",   "")
        metrics = tool_result.get("metrics", [])

        if not metrics:
            return

        with result_container:
            if title:
                st.markdown(f"**{title}**")

            chunk_size = 4
            for i in range(0, len(metrics), chunk_size):
                row     = metrics[i : i + chunk_size]
                columns = st.columns(len(row))
                for col, metric in zip(columns, row):
                    with col:
                        st.metric(
                            label       = metric.get("label", ""),
                            value       = metric.get("value", ""),
                            delta       = metric.get("delta"),
                            delta_color = metric.get("delta_color", "normal"),
                        )