import os
import streamlit as st
from typing import Any
from cmn.tools.renderer.bedrock_converse_tools_renderer import AbstractToolRenderer


class PptxToolRenderer(AbstractToolRenderer):

    @property
    def tool_name(self) -> str:
        return "create_pptx"

    def render(
        self,
        tool_args:        dict,
        tool_result:      Any,
        result_container,
    ) -> None:

        # ── Error guard ───────────────────────────────────────────────────────
        if "error" in tool_result:
            with result_container:
                st.warning(f"PPTX error: {tool_result['error']}")
            return

        if tool_result.get("status") != "pptx_ready":
            return

        filepath    = tool_result.get("filepath",     "")
        filename    = tool_result.get("filename",     "presentation.pptx")
        title       = tool_result.get("title",        "")
        brand       = tool_result.get("brand",        "")
        slide_count = tool_result.get("slide_count",  0)
        titles      = tool_result.get("slide_titles", [])
        warnings    = tool_result.get("warnings",     [])

        # ── File guard ────────────────────────────────────────────────────────
        if not filepath or not os.path.exists(filepath):
            with result_container:
                st.warning(
                    f"Presentation file not found: `{filepath}`. "
                    "It may have been cleaned up."
                )
            return

        with result_container:

            # ── Header ────────────────────────────────────────────────────────
            st.markdown(f"**📊 {title}**")
            st.caption(
                f"Brand: `{brand}` · {slide_count} slides · `{filename}`"
            )

            # ── Warnings ──────────────────────────────────────────────────────
            for w in warnings:
                st.warning(w)

            # ── Slide outline ─────────────────────────────────────────────────
            if titles:
                with st.expander("📋 Slide Outline", expanded=False):
                    for i, t in enumerate(titles, 1):
                        label = t if t else "*(untitled)*"
                        st.markdown(f"{i}. {label}")

            # ── Read file here — not in invoke() ──────────────────────────────
            with open(filepath, "rb") as f:
                pptx_bytes = f.read()

            st.download_button(
                label               = "⬇️ Download PowerPoint",
                data                = pptx_bytes,
                file_name           = filename,
                mime                = (
                    "application/vnd.openxmlformats-officedocument"
                    ".presentationml.presentation"
                ),
                use_container_width = True,
            )