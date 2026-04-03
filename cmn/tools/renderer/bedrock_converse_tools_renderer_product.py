import streamlit as st
from typing import Any
from cmn.tools.renderer.bedrock_converse_tools_renderer import AbstractToolRenderer


class ProductToolRenderer(AbstractToolRenderer):

    @property
    def tool_name(self) -> str:
        return "product_query"

    def render(self, tool_args: dict, tool_result: Any, result_container) -> None:
        with result_container:
            st.markdown(":blue[🗄️ **Generated SQL:**]")
            st.code(
                tool_args.get("sql", ""),
                language="sql",
                wrap_lines=True,
            )