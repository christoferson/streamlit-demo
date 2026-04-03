# cmn/tools/renderer/bedrock_converse_tools_renderer_pdf.py

import os
import streamlit as st
from typing import Any
from cmn.tools.renderer.bedrock_converse_tools_renderer import AbstractToolRenderer


class PdfToolRenderer(AbstractToolRenderer):

    @property
    def tool_name(self) -> str:
        return "create_pdf"

    def render(
        self,
        tool_args:        dict,
        tool_result:      Any,
        result_container,
    ) -> None:

        # ── Error guard ───────────────────────────────────────────────────────
        if "error" in tool_result:
            with result_container:
                st.warning(f"PDF error: {tool_result['error']}")
            return

        if tool_result.get("status") != "pdf_ready":
            return

        filepath  = tool_result.get("filepath",  "")
        filename  = tool_result.get("filename",  "report.pdf")
        title     = tool_result.get("title",     "")
        brand     = tool_result.get("brand",     "")
        pages     = tool_result.get("pages",     0)
        warnings  = tool_result.get("warnings",  [])

        # ── File guard ────────────────────────────────────────────────────────
        if not filepath or not os.path.exists(filepath):
            with result_container:
                st.warning(
                    f"PDF file not found: `{filepath}`. "
                    "It may have been cleaned up."
                )
            return

        # ── Load bytes — from file on first call, session_state on rerun ──────
        cache_key  = f"pdf_bytes_{filepath}"
        pdf_bytes  = st.session_state.get(cache_key)

        if pdf_bytes is None:
            with open(filepath, "rb") as f:
                pdf_bytes = f.read()
            st.session_state[cache_key] = pdf_bytes

        with result_container:

            # ── Header ────────────────────────────────────────────────────────
            st.markdown(f"**📄 {title}**")
            st.caption(
                f"Brand: `{brand}` · {pages} page{'s' if pages != 1 else ''} · `{filename}`"
            )

            # ── Warnings ──────────────────────────────────────────────────────
            for w in warnings:
                st.info(f"💡 {w}")

            # ── Download button ───────────────────────────────────────────────
            st.download_button(
                label               = "⬇️ Download PDF",
                data                = pdf_bytes,
                file_name           = filename,
                mime                = "application/pdf",
                use_container_width = True,
            )