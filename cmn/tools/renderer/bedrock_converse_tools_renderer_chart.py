import pandas as pd
import plotly.express as px
import streamlit as st
from typing import Any
from cmn.tools.renderer.bedrock_converse_tools_renderer import AbstractToolRenderer


class ChartToolRenderer(AbstractToolRenderer):

    @property
    def tool_name(self) -> str:
        return "render_chart"

    def render(self, tool_args: dict, tool_result: Any, result_container) -> None:

        data  = tool_args.get("data") or tool_result.get("data", [])
        x     = tool_args["x_label"]
        y     = tool_args["y_label"]
        title = tool_args["title"]
        ctype = tool_args["chart_type"]

        if not data:
            with result_container:
                st.warning("Chart error: no data provided.")
            return

        df = pd.DataFrame(data)

        if x not in df.columns:
            with result_container:
                st.warning(f"Chart error: x='{x}' not found. Got: {list(df.columns)}")
            return

        # ── Resolve color column ──────────────────────────────────────────────
        # 1. explicit color_series from tool_args (model declared it)
        # 2. explicit color_series from tool_result (echoed back by invoke)
        # 3. fallback — detect known series column names in the data
        color_col = (
            tool_args.get("color_series")
            or tool_result.get("color_series")
            or next(
                (c for c in ["series", "type", "category", "year"] if c in df.columns),
                None,
            )
        )

        x_order = df[x].unique().tolist()

        with result_container:
            st.markdown(f"**{title}**")

            if color_col and color_col in df.columns:
                # ── Long format — one row per x+series combination ────────────
                if ctype == "bar":
                    fig = px.bar(df, x=x, y=y, color=color_col, barmode="group")
                elif ctype == "line":
                    fig = px.line(df, x=x, y=y, color=color_col, markers=True)
                else:
                    fig = px.area(df, x=x, y=y, color=color_col)

            else:
                # ── Wide format — one column per series ───────────────────────
                chart_df  = df.set_index(x)
                y_columns = (
                    [y] if y in chart_df.columns
                    else chart_df.select_dtypes(include="number").columns.tolist()
                )
                plot_df = chart_df[y_columns].reset_index()

                if ctype == "bar":
                    fig = px.bar(plot_df, x=x, y=y_columns, barmode="group")
                elif ctype == "line":
                    fig = px.line(plot_df, x=x, y=y_columns, markers=True)
                else:
                    fig = px.area(plot_df, x=x, y=y_columns)

            fig.update_xaxes(categoryorder="array", categoryarray=x_order)
            st.plotly_chart(fig, width="content")