# cmn/tools/tool/pdf/builder.py

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
import logging
import os
from typing import Optional

from reportlab.lib.colors import Color, HexColor
from reportlab.lib.units import inch
from reportlab.pdfgen.canvas import Canvas

logger = logging.getLogger(__name__)


################################################################################
# SECTION: PdfDrawState
################################################################################

@dataclass
class PdfDrawState:
    """
    Tracks mutable drawing position as sections are rendered top-to-bottom.

    y           current top of next element in points from bottom of page.
                reportlab origin is bottom-left, y increases upward.
                We track from top — so y starts at (page_height - margin_top)
                and decreases as content is added.

    page_num    current page number, incremented on each new page.

    All dimensions stored in points (1 inch = 72 pts).
    """

    y:             float = 0.0
    page_num:      int   = 1

    # Set once at build time — never change after init
    page_width:    float = 0.0
    page_height:   float = 0.0
    margin_top:    float = 0.0
    margin_bottom: float = 0.0
    margin_left:   float = 0.0
    margin_right:  float = 0.0

    @property
    def content_width(self) -> float:
        """Usable width between left and right margins."""
        return self.page_width - self.margin_left - self.margin_right

    @property
    def content_top(self) -> float:
        """Y position of top of content area (below top margin)."""
        return self.page_height - self.margin_top

    @property
    def content_bottom(self) -> float:
        """Y position of bottom of content area (above bottom margin)."""
        return self.margin_bottom

    def needs_page_break(self, required_height: float) -> bool:
        """
        True if remaining vertical space is less than required_height.
        Caller should trigger a new page before drawing.
        """
        return self.y - required_height < self.content_bottom

    def advance(self, height: float):
        """Move y down by height points."""
        self.y -= height

    def reset_to_top(self):
        """Reset y to top of content area — called after new page."""
        self.y = self.content_top


################################################################################
# SECTION: AbstractPdfSectionRenderer
################################################################################

class AbstractPdfSectionRenderer(ABC):
    """
    Base class for all PDF section renderers.

    Each renderer owns exactly two things:
      1. section_type  — matches section_type string from LLM input
      2. render()      — draws onto canvas, updates state.y

    Renderers must:
      - Check state.needs_page_break() before drawing large elements
      - Call _new_page() if a page break is needed
      - Call state.advance() after drawing each element
      - Never store canvas or state as instance variables
        (canvas is passed per render call — stateless renderers)

    Renderers must NOT:
      - Import streamlit
      - Know about tool_args or tool_result
      - Call other renderers directly
    """

    @property
    @abstractmethod
    def section_type(self) -> str:
        """
        Must match section_type string exactly as sent by LLM.
        Example: "heading", "text", "table", "chart"
        """

    @abstractmethod
    def render(
        self,
        canvas:  Canvas,
        section: dict,
        brand,              # PdfBrandGuidelines — not imported here
        state:   PdfDrawState,
        on_new_page,        # Callable[[Canvas, PdfBrandGuidelines, PdfDrawState], None]
    ) -> None:
        """
        Draw this section onto canvas.
        Updates state.y after drawing.

        canvas      reportlab Canvas object
        section     dict from LLM — section definition
        brand       PdfBrandGuidelines instance
        state       PdfDrawState — mutable, update state.y as you draw
        on_new_page callable — call this when a page break is needed
                    signature: on_new_page(canvas, brand, state)
        """

    # ── Shared drawing helpers available to all subclasses ────────────────────

    @staticmethod
    def _set_fill_color(canvas: Canvas, hex_color: str):
        """Set canvas fill color from hex string."""
        canvas.setFillColor(HexColor(hex_color))

    @staticmethod
    def _set_stroke_color(canvas: Canvas, hex_color: str):
        """Set canvas stroke color from hex string."""
        canvas.setStrokeColor(HexColor(hex_color))

    @staticmethod
    def _draw_rect_filled(
        canvas:    Canvas,
        x:         float,
        y:         float,
        width:     float,
        height:    float,
        hex_color: str,
    ):
        """Draw a filled rectangle. y is bottom-left corner in pts."""
        canvas.setFillColor(HexColor(hex_color))
        canvas.rect(x, y, width, height, stroke=0, fill=1)

    @staticmethod
    def _draw_line(
        canvas:    Canvas,
        x1:        float,
        y1:        float,
        x2:        float,
        y2:        float,
        hex_color: str,
        width:     float = 0.5,
    ):
        """Draw a horizontal or vertical line."""
        canvas.setStrokeColor(HexColor(hex_color))
        canvas.setLineWidth(width)
        canvas.line(x1, y1, x2, y2)

    @staticmethod
    def _draw_text(
        canvas:    Canvas,
        text:      str,
        x:         float,
        y:         float,
        font:      str,
        size:      int,
        hex_color: str,
    ):
        """Draw a single line of text. y is baseline in pts."""
        canvas.setFillColor(HexColor(hex_color))
        canvas.setFont(font, size)
        canvas.drawString(x, y, text)

    @staticmethod
    def _draw_text_right(
        canvas:    Canvas,
        text:      str,
        x:         float,
        y:         float,
        font:      str,
        size:      int,
        hex_color: str,
    ):
        """Draw right-aligned text. x is right edge."""
        canvas.setFillColor(HexColor(hex_color))
        canvas.setFont(font, size)
        canvas.drawRightString(x, y, text)

    @staticmethod
    def _draw_text_centered(
        canvas:    Canvas,
        text:      str,
        x:         float,
        y:         float,
        font:      str,
        size:      int,
        hex_color: str,
    ):
        """Draw centered text. x is center point."""
        canvas.setFillColor(HexColor(hex_color))
        canvas.setFont(font, size)
        canvas.drawCentredString(x, y, text)

    @staticmethod
    def _wrap_text(
        text:       str,
        font:       str,
        size:       int,
        max_width:  float,
    ) -> list[str]:
        """
        Wrap text into lines that fit within max_width points.
        Uses reportlab stringWidth for accurate measurement.
        Returns list of line strings.
        """
        from reportlab.pdfbase.pdfmetrics import stringWidth

        words = text.split()
        lines = []
        current = ""

        for word in words:
            test = f"{current} {word}".strip()
            if stringWidth(test, font, size) <= max_width:
                current = test
            else:
                if current:
                    lines.append(current)
                current = word

        if current:
            lines.append(current)

        return lines or [""]

    @staticmethod
    def _line_height(size: int, leading: float = 1.4) -> float:
        """
        Returns line height in points.
        leading multiplier controls spacing between lines.
        Default 1.4 = 40% extra space above font size.
        """
        return size * leading
    
################################################################################
# SECTION: PdfCoverSectionRenderer
################################################################################

class PdfCoverSectionRenderer(AbstractPdfSectionRenderer):
    """
    Full cover page.
    Fills background with brand primary color.
    Draws title, subtitle, author, date centered on page.
    Always occupies its own page — triggers page break after.
    """

    @property
    def section_type(self) -> str:
        return "cover"

    def render(self, canvas, section, brand, state, on_new_page):
        W = state.page_width
        H = state.page_height

        # ── Full page background ──────────────────────────────────────────────
        self._draw_rect_filled(
            canvas,
            x         = 0,
            y         = 0,
            width     = W,
            height    = H,
            hex_color = brand.colors.primary,
        )

        # ── Accent bar at bottom ──────────────────────────────────────────────
        bar_h = brand.rules.accent_bar_height * inch
        self._draw_rect_filled(
            canvas,
            x         = 0,
            y         = 0,
            width     = W,
            height    = bar_h,
            hex_color = brand.colors.accent,
        )

        # ── Title ─────────────────────────────────────────────────────────────
        title = section.get("title", "")
        if title:
            self._draw_text_centered(
                canvas,
                text      = title,
                x         = W / 2,
                y         = H * 0.55,
                font      = brand.fonts.heading,
                size      = brand.fonts.cover_title_size,
                hex_color = brand.colors.text_light,
            )

        # ── Subtitle ──────────────────────────────────────────────────────────
        subtitle = section.get("subtitle", "")
        if subtitle:
            self._draw_text_centered(
                canvas,
                text      = subtitle,
                x         = W / 2,
                y         = H * 0.47,
                font      = brand.fonts.body,
                size      = brand.fonts.h2_size,
                hex_color = brand.colors.accent,
            )

        # ── Divider line ──────────────────────────────────────────────────────
        self._draw_line(
            canvas,
            x1        = W * 0.2,
            y1        = H * 0.44,
            x2        = W * 0.8,
            y2        = H * 0.44,
            hex_color = brand.colors.accent,
            width     = 0.5,
        )

        # ── Author ────────────────────────────────────────────────────────────
        author = section.get("author", "")
        if author:
            self._draw_text_centered(
                canvas,
                text      = author,
                x         = W / 2,
                y         = H * 0.40,
                font      = brand.fonts.body,
                size      = brand.fonts.body_size,
                hex_color = brand.colors.text_light,
            )

        # ── Date ──────────────────────────────────────────────────────────────
        date_str = section.get("date") or datetime.now().strftime("%B %d, %Y")
        self._draw_text_centered(
            canvas,
            text      = date_str,
            x         = W / 2,
            y         = H * 0.36,
            font      = brand.fonts.body,
            size      = brand.fonts.caption_size,
            hex_color = brand.colors.text_muted,
        )

        # ── Cover always ends with a new page ─────────────────────────────────
        canvas.showPage()
        state.page_num += 1
        state.reset_to_top()


################################################################################
# SECTION: PdfHeadingSectionRenderer
################################################################################

class PdfHeadingSectionRenderer(AbstractPdfSectionRenderer):
    """
    Section heading — level 1 or level 2.

    level 1:  larger font, primary color, underline rule below
    level 2:  smaller font, secondary color, no rule
    """

    @property
    def section_type(self) -> str:
        return "heading"

    def render(self, canvas, section, brand, state, on_new_page):
        text  = section.get("text", "")
        level = section.get("level", 1)

        if not text:
            return

        # ── Pick style by level ───────────────────────────────────────────────
        if level == 1:
            font      = brand.fonts.heading
            size      = brand.fonts.h1_size
            hex_color = brand.colors.primary
            draw_rule = True
            space_before = brand.rules.section_space_before * inch * 1.5
        else:
            font      = brand.fonts.heading
            size      = brand.fonts.h2_size
            hex_color = brand.colors.secondary
            draw_rule = False
            space_before = brand.rules.section_space_before * inch

        line_h    = self._line_height(size)
        rule_h    = 1.5 if draw_rule else 0
        gap       = 4   if draw_rule else 0
        total_h   = space_before + line_h + gap + rule_h

        if state.needs_page_break(total_h):
            on_new_page(canvas, brand, state)

        # ── Space before ──────────────────────────────────────────────────────
        state.advance(space_before)

        # ── Heading text ──────────────────────────────────────────────────────
        self._draw_text(
            canvas,
            text      = text,
            x         = state.margin_left,
            y         = state.y - line_h,
            font      = font,
            size      = size,
            hex_color = hex_color,
        )
        state.advance(line_h)

        # ── Rule under h1 ─────────────────────────────────────────────────────
        if draw_rule:
            state.advance(gap)
            self._draw_line(
                canvas,
                x1        = state.margin_left,
                y1        = state.y,
                x2        = state.margin_left + state.content_width,
                y2        = state.y,
                hex_color = brand.colors.primary,
                width     = 1.0,
            )
            state.advance(rule_h + 4)


################################################################################
# SECTION: PdfTextSectionRenderer
################################################################################

class PdfTextSectionRenderer(AbstractPdfSectionRenderer):
    """
    Body paragraph — wraps text to content width.
    Handles multi-line text with correct line spacing.
    """

    @property
    def section_type(self) -> str:
        return "text"

    def render(self, canvas, section, brand, state, on_new_page):
        text = section.get("text", "")

        if not text:
            return

        font   = brand.fonts.body
        size   = brand.fonts.body_size
        line_h = self._line_height(size)

        # ── Wrap text to content width ────────────────────────────────────────
        lines = self._wrap_text(text, font, size, state.content_width)

        # ── Space before paragraph ────────────────────────────────────────────
        space_before = brand.rules.section_space_before * inch * 0.5
        state.advance(space_before)

        for line in lines:
            if state.needs_page_break(line_h):
                on_new_page(canvas, brand, state)

            self._draw_text(
                canvas,
                text      = line,
                x         = state.margin_left,
                y         = state.y - line_h,
                font      = font,
                size      = size,
                hex_color = brand.colors.text_dark,
            )
            state.advance(line_h)

        # ── Space after paragraph ─────────────────────────────────────────────
        state.advance(space_before)


################################################################################
# SECTION: PdfBulletsSectionRenderer
################################################################################

class PdfBulletsSectionRenderer(AbstractPdfSectionRenderer):
    """
    Bullet list.
    Each item wrapped to content width minus indent.
    Bullet marker drawn in accent color.
    Text drawn in text_dark.
    """

    _BULLET_CHAR = "▪"

    @property
    def section_type(self) -> str:
        return "bullets"

    def render(self, canvas, section, brand, state, on_new_page):
        items = section.get("items", [])

        if not items:
            return

        font         = brand.fonts.body
        size         = brand.fonts.body_size
        line_h       = self._line_height(size)
        indent       = brand.rules.bullet_indent * inch
        text_x       = state.margin_left + indent
        text_width   = state.content_width - indent
        space_before = brand.rules.section_space_before * inch * 0.5

        state.advance(space_before)

        for item in items:
            # ── Wrap bullet text ──────────────────────────────────────────────
            lines = self._wrap_text(item, font, size, text_width)

            # ── Check space for at least first line ───────────────────────────
            if state.needs_page_break(line_h):
                on_new_page(canvas, brand, state)

            # ── Bullet marker on first line only ──────────────────────────────
            self._draw_text(
                canvas,
                text      = self._BULLET_CHAR,
                x         = state.margin_left,
                y         = state.y - line_h,
                font      = font,
                size      = size,
                hex_color = brand.colors.accent,
            )

            # ── First line text ───────────────────────────────────────────────
            self._draw_text(
                canvas,
                text      = lines[0],
                x         = text_x,
                y         = state.y - line_h,
                font      = font,
                size      = size,
                hex_color = brand.colors.text_dark,
            )
            state.advance(line_h)

            # ── Continuation lines (wrapped) ──────────────────────────────────
            for continuation in lines[1:]:
                if state.needs_page_break(line_h):
                    on_new_page(canvas, brand, state)

                self._draw_text(
                    canvas,
                    text      = continuation,
                    x         = text_x,
                    y         = state.y - line_h,
                    font      = font,
                    size      = size,
                    hex_color = brand.colors.text_dark,
                )
                state.advance(line_h)

            # ── Small gap between bullet items ────────────────────────────────
            state.advance(3)

        state.advance(space_before)

################################################################################
# SECTION: PdfTableSectionRenderer
################################################################################

class PdfTableSectionRenderer(AbstractPdfSectionRenderer):
    """
    Data table with branded header row and alternating row colors.

    section input:
      headers: ["Month", "Revenue", "Units", "Margin"]
      rows:    [["January", "$213,000", "1,035", "45%"], ...]
    """

    @property
    def section_type(self) -> str:
        return "table"

    def render(self, canvas, section, brand, state, on_new_page):
        headers = section.get("headers", [])
        rows    = section.get("rows",    [])

        if not headers and not rows:
            return

        font         = brand.fonts.body
        header_font  = brand.fonts.heading
        size         = brand.fonts.body_size
        caption_size = brand.fonts.caption_size
        row_h        = brand.rules.table_row_height * inch
        space_before = brand.rules.section_space_before * inch

        n_cols       = max(len(headers), max((len(r) for r in rows), default=0))
        if n_cols == 0:
            return

        col_w        = state.content_width / n_cols
        x_start      = state.margin_left

        state.advance(space_before)

        # ── Header row ────────────────────────────────────────────────────────
        if headers:
            if state.needs_page_break(row_h):
                on_new_page(canvas, brand, state)

            # Header background
            self._draw_rect_filled(
                canvas,
                x         = x_start,
                y         = state.y - row_h,
                width     = state.content_width,
                height    = row_h,
                hex_color = brand.colors.primary,
            )

            # Header text
            for i, header in enumerate(headers):
                cell_x = x_start + (i * col_w) + 6
                self._draw_text(
                    canvas,
                    text      = str(header),
                    x         = cell_x,
                    y         = state.y - row_h + (row_h - size) / 2,
                    font      = header_font,
                    size      = caption_size,
                    hex_color = brand.colors.text_light,
                )

            state.advance(row_h)

        # ── Data rows ─────────────────────────────────────────────────────────
        for row_idx, row in enumerate(rows):
            if state.needs_page_break(row_h):
                on_new_page(canvas, brand, state)

                # Repeat header on new page
                if headers:
                    self._draw_rect_filled(
                        canvas,
                        x         = x_start,
                        y         = state.y - row_h,
                        width     = state.content_width,
                        height    = row_h,
                        hex_color = brand.colors.primary,
                    )
                    for i, header in enumerate(headers):
                        cell_x = x_start + (i * col_w) + 6
                        self._draw_text(
                            canvas,
                            text      = str(header),
                            x         = cell_x,
                            y         = state.y - row_h + (row_h - size) / 2,
                            font      = header_font,
                            size      = caption_size,
                            hex_color = brand.colors.text_light,
                        )
                    state.advance(row_h)

            # ── Alternating row background ────────────────────────────────────
            bg_color = brand.colors.surface if row_idx % 2 == 0 else brand.colors.background
            self._draw_rect_filled(
                canvas,
                x         = x_start,
                y         = state.y - row_h,
                width     = state.content_width,
                height    = row_h,
                hex_color = bg_color,
            )

            # ── Row border line ───────────────────────────────────────────────
            self._draw_line(
                canvas,
                x1        = x_start,
                y1        = state.y - row_h,
                x2        = x_start + state.content_width,
                y2        = state.y - row_h,
                hex_color = brand.colors.text_muted,
                width     = 0.3,
            )

            # ── Cell text ─────────────────────────────────────────────────────
            for col_idx, cell in enumerate(row):
                cell_x = x_start + (col_idx * col_w) + 6
                cell_text = str(cell)

                # Truncate if too wide for column
                from reportlab.pdfbase.pdfmetrics import stringWidth
                max_cell_w = col_w - 12
                while (
                    len(cell_text) > 1
                    and stringWidth(cell_text, font, caption_size) > max_cell_w
                ):
                    cell_text = cell_text[:-2] + "…"

                self._draw_text(
                    canvas,
                    text      = cell_text,
                    x         = cell_x,
                    y         = state.y - row_h + (row_h - caption_size) / 2,
                    font      = font,
                    size      = caption_size,
                    hex_color = brand.colors.text_dark,
                )

            state.advance(row_h)

        # ── Outer border ──────────────────────────────────────────────────────
        # drawn last so it sits on top of row fills
        self._draw_line(
            canvas,
            x1        = x_start,
            y1        = state.y,
            x2        = x_start + state.content_width,
            y2        = state.y,
            hex_color = brand.colors.primary,
            width     = 0.5,
        )

        # ── Column dividers ───────────────────────────────────────────────────
        table_top    = state.y + (row_h * (len(rows) + (1 if headers else 0)))
        table_bottom = state.y

        for i in range(1, n_cols):
            col_x = x_start + (i * col_w)
            self._draw_line(
                canvas,
                x1        = col_x,
                y1        = table_bottom,
                x2        = col_x,
                y2        = table_top,
                hex_color = brand.colors.text_muted,
                width     = 0.3,
            )

        state.advance(space_before)


################################################################################
# SECTION: PdfMetricRowSectionRenderer
################################################################################

class PdfMetricRowSectionRenderer(AbstractPdfSectionRenderer):
    """
    KPI metric boxes in a horizontal row.
    Each metric has label, value, and optional delta.

    section input:
      metrics: [
        {"label": "Total Revenue", "value": "$3.13M", "delta": "+5%"},
        {"label": "Units Sold",    "value": "14,905", "delta": "-4%"},
        {"label": "Gross Profit",  "value": "$1.38M"},
        {"label": "Margin",        "value": "45%",    "delta": "0%"},
      ]

    Layout:
      ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
      │ Total Revenue│ │  Units Sold  │ │ Gross Profit │ │    Margin    │
      │   $3.13M     │ │   14,905     │ │   $1.38M     │ │    45%       │
      │    +5%  ▲    │ │    -4%  ▼    │ │    +3%  ▲    │ │    0%   ─    │
      └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘
    """

    # Delta color rules
    _POSITIVE_PREFIXES = ("+",)
    _NEGATIVE_PREFIXES = ("-",)

    @property
    def section_type(self) -> str:
        return "metric_row"

    def render(self, canvas, section, brand, state, on_new_page):
        metrics = section.get("metrics", [])

        if not metrics:
            return

        space_before = brand.rules.section_space_before * inch
        box_h        = 0.85 * inch
        gap          = 8                          # pts between boxes
        n            = len(metrics)
        box_w        = (state.content_width - (gap * (n - 1))) / n
        total_h      = space_before + box_h + space_before

        if state.needs_page_break(total_h):
            on_new_page(canvas, brand, state)

        state.advance(space_before)

        for i, metric in enumerate(metrics):
            x = state.margin_left + i * (box_w + gap)
            y = state.y - box_h

            label = metric.get("label", "")
            value = metric.get("value", "")
            delta = metric.get("delta", "")

            # ── Box background ────────────────────────────────────────────────
            self._draw_rect_filled(
                canvas,
                x         = x,
                y         = y,
                width     = box_w,
                height    = box_h,
                hex_color = brand.colors.surface,
            )

            # ── Left accent bar ───────────────────────────────────────────────
            self._draw_rect_filled(
                canvas,
                x         = x,
                y         = y,
                width     = 3,
                height    = box_h,
                hex_color = brand.colors.primary,
            )

            # ── Label ─────────────────────────────────────────────────────────
            self._draw_text(
                canvas,
                text      = label,
                x         = x + 10,
                y         = y + box_h - 16,
                font      = brand.fonts.body,
                size      = brand.fonts.caption_size,
                hex_color = brand.colors.text_muted,
            )

            # ── Value ─────────────────────────────────────────────────────────
            self._draw_text(
                canvas,
                text      = value,
                x         = x + 10,
                y         = y + box_h - 36,
                font      = brand.fonts.heading,
                size      = brand.fonts.h2_size,
                hex_color = brand.colors.primary,
            )

            # ── Delta ─────────────────────────────────────────────────────────
            if delta:
                delta_color, arrow = self._delta_style(delta, brand)
                self._draw_text(
                    canvas,
                    text      = f"{delta} {arrow}",
                    x         = x + 10,
                    y         = y + 8,
                    font      = brand.fonts.body,
                    size      = brand.fonts.caption_size,
                    hex_color = delta_color,
                )

        state.advance(box_h)
        state.advance(space_before)

    def _delta_style(self, delta: str, brand) -> tuple:
        """
        Returns (hex_color, arrow_char) based on delta prefix.
        + → success green, ▲
        - → danger red,    ▼
        0 or neutral → muted, ─
        """
        stripped = delta.strip()
        if stripped.startswith(self._POSITIVE_PREFIXES):
            return brand.colors.success, "▲"
        if stripped.startswith(self._NEGATIVE_PREFIXES):
            return brand.colors.danger,  "▼"
        return brand.colors.text_muted, "─"


################################################################################
# SECTION: PdfPageBreakSectionRenderer
################################################################################

class PdfPageBreakSectionRenderer(AbstractPdfSectionRenderer):
    """
    Explicit page break.
    LLM can insert this to force content onto a new page.
    """

    @property
    def section_type(self) -> str:
        return "page_break"

    def render(self, canvas, section, brand, state, on_new_page):
        on_new_page(canvas, brand, state)

################################################################################
# SECTION: PdfChartSectionRenderer
################################################################################

class PdfChartSectionRenderer(AbstractPdfSectionRenderer):
    """
    Renders native vector charts using reportlab.graphics.
    No matplotlib. No PNG embedding. Pure reportlab.

    Supported chart types:
      bar   → VerticalBarChart
      line  → LinePlot
      pie   → Pie

    chart_data contract (same as PPT tool — domain agnostic):
      Single series:  [{"label": "Jan", "value": 213000}, ...]
      Multi-series:   [{"label": "Jan", "value": 213000, "series": "2024"}, ...]
    """

    @property
    def section_type(self) -> str:
        return "chart"

    def render(self, canvas, section, brand, state, on_new_page):
        chart_data_raw = section.get("chart_data",  [])
        chart_type     = section.get("chart_type",  "bar")
        title          = section.get("title",       "")
        x_label        = section.get("x_label",     "")
        y_label        = section.get("y_label",     "")

        if not chart_data_raw:
            logger.warning("PdfChartSectionRenderer: chart_data empty — skipped")
            return

        chart_h      = brand.rules.chart_height_inches * inch
        title_h      = (self._line_height(brand.fonts.body_size) + 6) if title else 0
        space_before = brand.rules.section_space_before * inch
        total_h      = space_before + title_h + chart_h + space_before

        if state.needs_page_break(total_h):
            on_new_page(canvas, brand, state)

        state.advance(space_before)

        # ── Optional chart title ──────────────────────────────────────────────
        if title:
            self._draw_text(
                canvas,
                text      = title,
                x         = state.margin_left,
                y         = state.y - self._line_height(brand.fonts.body_size),
                font      = brand.fonts.heading,
                size      = brand.fonts.body_size,
                hex_color = brand.colors.primary,
            )
            state.advance(title_h)

        # ── Parse chart data ──────────────────────────────────────────────────
        has_series = any("series" in row for row in chart_data_raw)

        if has_series:
            labels, series_map = self._parse_multi(chart_data_raw)
        else:
            labels, series_map = self._parse_single(chart_data_raw)

        # ── Dispatch to chart builder ─────────────────────────────────────────
        chart_x = state.margin_left
        chart_y = state.y - chart_h     # reportlab y = bottom-left

        try:
            if chart_type == "bar":
                self._draw_bar_chart(
                    canvas, chart_x, chart_y,
                    state.content_width, chart_h,
                    labels, series_map,
                    x_label, y_label, brand,
                )
            elif chart_type == "line":
                self._draw_line_chart(
                    canvas, chart_x, chart_y,
                    state.content_width, chart_h,
                    labels, series_map,
                    x_label, y_label, brand,
                )
            elif chart_type == "pie":
                self._draw_pie_chart(
                    canvas, chart_x, chart_y,
                    state.content_width, chart_h,
                    labels, series_map,
                    brand,
                )
            else:
                logger.warning("Unknown chart_type '%s' — falling back to bar", chart_type)
                self._draw_bar_chart(
                    canvas, chart_x, chart_y,
                    state.content_width, chart_h,
                    labels, series_map,
                    x_label, y_label, brand,
                )
        except Exception as exc:
            logger.exception("PdfChartSectionRenderer: draw failed: %s", exc)

        state.advance(chart_h)
        state.advance(space_before)

    # ── Data parsers ──────────────────────────────────────────────────────────

    @staticmethod
    def _parse_single(data: list) -> tuple:
        """
        Returns (labels, series_map) for single-series data.
        series_map = {"Value": [v1, v2, ...]}
        """
        labels     = [str(row["label"])   for row in data]
        values     = [float(row["value"]) for row in data]
        series_map = {"Value": values}
        return labels, series_map

    @staticmethod
    def _parse_multi(data: list) -> tuple:
        """
        Returns (labels, series_map) for multi-series data.
        series_map = {"2023": [v1, v2, ...], "2024": [v1, v2, ...]}
        Preserves label order. Fills missing values with 0.
        """
        from collections import defaultdict

        label_order = []
        raw_map     = defaultdict(dict)

        for row in data:
            label  = str(row["label"])
            value  = float(row["value"])
            series = str(row.get("series", "Value"))

            if label not in label_order:
                label_order.append(label)
            raw_map[series][label] = value

        series_map = {
            name: [raw_map[name].get(lbl, 0) for lbl in label_order]
            for name in raw_map
        }
        return label_order, series_map

    # ── Bar chart ─────────────────────────────────────────────────────────────

    def _draw_bar_chart(
        self,
        canvas,
        x: float, y: float,
        width: float, height: float,
        labels: list, series_map: dict,
        x_label: str, y_label: str,
        brand,
    ):
        from reportlab.graphics.shapes import Drawing, String
        from reportlab.graphics.charts.barcharts import VerticalBarChart
        from reportlab.graphics import renderPDF
        from reportlab.lib.colors import HexColor

        drawing = Drawing(width, height)
        chart   = VerticalBarChart()

        # ── Position inside drawing ───────────────────────────────────────────
        left_pad   = 50 if y_label else 30
        bottom_pad = 40 if x_label else 25
        top_pad    = 10
        right_pad  = 20

        chart.x      = left_pad
        chart.y      = bottom_pad
        chart.width  = width  - left_pad  - right_pad
        chart.height = height - bottom_pad - top_pad

        # ── Data ──────────────────────────────────────────────────────────────
        series_names = list(series_map.keys())
        chart.data   = [series_map[name] for name in series_names]

        # ── Category labels ───────────────────────────────────────────────────
        chart.categoryAxis.categoryNames = labels
        chart.categoryAxis.labels.angle  = 45 if len(labels) > 6 else 0
        chart.categoryAxis.labels.dy     = -10 if len(labels) > 6 else -5
        chart.categoryAxis.labels.fontSize   = brand.fonts.caption_size - 1
        chart.categoryAxis.labels.fontName   = brand.fonts.body
        chart.categoryAxis.labels.fillColor  = HexColor(brand.colors.text_dark)
        chart.categoryAxis.strokeColor       = HexColor(brand.colors.text_muted)

        # ── Value axis ────────────────────────────────────────────────────────
        chart.valueAxis.labels.fontSize  = brand.fonts.caption_size - 1
        chart.valueAxis.labels.fontName  = brand.fonts.body
        chart.valueAxis.labels.fillColor = HexColor(brand.colors.text_dark)
        chart.valueAxis.strokeColor      = HexColor(brand.colors.text_muted)
        chart.valueAxis.gridStrokeColor  = HexColor(brand.colors.text_muted)
        chart.valueAxis.gridStrokeDashArray = [2, 4]

        # ── Series colors ─────────────────────────────────────────────────────
        palette = brand.colors.chart_palette
        for i in range(len(series_names)):
            color = HexColor(palette[i % len(palette)])
            chart.bars[i].fillColor   = color
            chart.bars[i].strokeColor = color

        chart.bars.strokeWidth = 0

        # ── Axis labels ───────────────────────────────────────────────────────
        if x_label:
            label = String(
                left_pad + chart.width / 2,
                5,
                x_label,
                fontSize  = brand.fonts.caption_size,
                fontName  = brand.fonts.body,
                fillColor = HexColor(brand.colors.text_muted),
                textAnchor = "middle",
            )
            drawing.add(label)

        if y_label:
            from reportlab.graphics.shapes import Group
            from reportlab.graphics.transform import translate, rotate
            label = String(
                10,
                bottom_pad + chart.height / 2,
                y_label,
                fontSize   = brand.fonts.caption_size,
                fontName   = brand.fonts.body,
                fillColor  = HexColor(brand.colors.text_muted),
                textAnchor = "middle",
            )
            drawing.add(label)

        # ── Legend for multi-series ───────────────────────────────────────────
        if len(series_names) > 1:
            self._add_legend(drawing, series_names, palette, brand,
                             width, height - top_pad - 10)

        drawing.add(chart)
        renderPDF.draw(drawing, canvas, x, y)

    # ── Line chart ────────────────────────────────────────────────────────────

    def _draw_line_chart(
        self,
        canvas,
        x: float, y: float,
        width: float, height: float,
        labels: list, series_map: dict,
        x_label: str, y_label: str,
        brand,
    ):
        from reportlab.graphics.shapes import Drawing, String, PolyLine, Circle
        from reportlab.graphics.charts.lineplots import LinePlot
        from reportlab.graphics import renderPDF
        from reportlab.lib.colors import HexColor

        drawing = Drawing(width, height)

        left_pad   = 55 if y_label else 35
        bottom_pad = 45 if x_label else 30
        top_pad    = 15
        right_pad  = 20

        chart_w = width  - left_pad  - right_pad
        chart_h = height - bottom_pad - top_pad

        series_names = list(series_map.keys())
        palette      = brand.colors.chart_palette
        n_points     = len(labels)

        if n_points < 2:
            logger.warning("Line chart needs at least 2 data points")
            return

        # ── Compute scale ─────────────────────────────────────────────────────
        all_values = [v for vals in series_map.values() for v in vals]
        min_val    = min(all_values) * 0.9
        max_val    = max(all_values) * 1.1
        val_range  = max_val - min_val or 1

        def _to_px(val):
            return bottom_pad + ((val - min_val) / val_range) * chart_h

        def _to_x(idx):
            return left_pad + (idx / (n_points - 1)) * chart_w

        # ── Grid lines ────────────────────────────────────────────────────────
        from reportlab.graphics.shapes import Line
        n_grid = 5
        for i in range(n_grid + 1):
            gv    = min_val + (i / n_grid) * val_range
            gy    = _to_px(gv)
            gline = Line(
                left_pad, gy,
                left_pad + chart_w, gy,
            )
            gline.strokeColor     = HexColor(brand.colors.text_muted)
            gline.strokeWidth     = 0.3
            gline.strokeDashArray = [2, 4]
            drawing.add(gline)

            # Y axis tick label
            label_str = self._format_value(gv)
            drawing.add(String(
                left_pad - 4, gy - 3,
                label_str,
                fontSize   = brand.fonts.caption_size - 2,
                fontName   = brand.fonts.body,
                fillColor  = HexColor(brand.colors.text_dark),
                textAnchor = "end",
            ))

        # ── X axis labels ─────────────────────────────────────────────────────
        step = max(1, n_points // 8)    # show at most 8 labels
        for i, lbl in enumerate(labels):
            if i % step == 0 or i == n_points - 1:
                lx = _to_x(i)
                drawing.add(String(
                    lx, bottom_pad - 12,
                    str(lbl),
                    fontSize   = brand.fonts.caption_size - 2,
                    fontName   = brand.fonts.body,
                    fillColor  = HexColor(brand.colors.text_dark),
                    textAnchor = "middle",
                ))

        # ── Axes ──────────────────────────────────────────────────────────────
        x_axis = Line(left_pad, bottom_pad, left_pad + chart_w, bottom_pad)
        x_axis.strokeColor = HexColor(brand.colors.text_muted)
        x_axis.strokeWidth = 0.5
        drawing.add(x_axis)

        y_axis = Line(left_pad, bottom_pad, left_pad, bottom_pad + chart_h)
        y_axis.strokeColor = HexColor(brand.colors.text_muted)
        y_axis.strokeWidth = 0.5
        drawing.add(y_axis)

        # ── Series lines + markers ────────────────────────────────────────────
        from reportlab.graphics.shapes import Circle
        for s_idx, name in enumerate(series_names):
            values    = series_map[name]
            color     = HexColor(palette[s_idx % len(palette)])
            points    = [(i, v) for i, v in enumerate(values)]

            # Line segments
            for i in range(len(points) - 1):
                seg = Line(
                    _to_x(points[i][0]),   _to_px(points[i][1]),
                    _to_x(points[i+1][0]), _to_px(points[i+1][1]),
                )
                seg.strokeColor = color
                seg.strokeWidth = 1.5
                drawing.add(seg)

            # Markers
            for i, v in enumerate(values):
                dot = Circle(_to_x(i), _to_px(v), 3)
                dot.fillColor   = color
                dot.strokeColor = color
                dot.strokeWidth = 0
                drawing.add(dot)

        # ── Axis labels ───────────────────────────────────────────────────────
        if x_label:
            drawing.add(String(
                left_pad + chart_w / 2, 5,
                x_label,
                fontSize   = brand.fonts.caption_size,
                fontName   = brand.fonts.body,
                fillColor  = HexColor(brand.colors.text_muted),
                textAnchor = "middle",
            ))

        if y_label:
            drawing.add(String(
                8,
                bottom_pad + chart_h / 2,
                y_label,
                fontSize   = brand.fonts.caption_size,
                fontName   = brand.fonts.body,
                fillColor  = HexColor(brand.colors.text_muted),
                textAnchor = "middle",
            ))

        # ── Legend for multi-series ───────────────────────────────────────────
        if len(series_names) > 1:
            self._add_legend(drawing, series_names, palette, brand,
                             width, height - top_pad)

        renderPDF.draw(drawing, canvas, x, y)

    # ── Pie chart ─────────────────────────────────────────────────────────────

    def _draw_pie_chart(
        self,
        canvas,
        x: float, y: float,
        width: float, height: float,
        labels: list, series_map: dict,
        brand,
    ):
        from reportlab.graphics.shapes import Drawing, String
        from reportlab.graphics.charts.piecharts import Pie
        from reportlab.graphics import renderPDF
        from reportlab.lib.colors import HexColor

        drawing = Drawing(width, height)
        chart   = Pie()

        # ── Use first series only for pie ─────────────────────────────────────
        series_name = list(series_map.keys())[0]
        values      = series_map[series_name]
        palette     = brand.colors.chart_palette

        chart.x      = width  * 0.15
        chart.y      = height * 0.1
        chart.width  = min(width * 0.55, height * 0.8)
        chart.height = chart.width
        chart.data   = values
        chart.labels = [
            f"{lbl} ({val / sum(values) * 100:.1f}%)"
            for lbl, val in zip(labels, values)
        ]

        # ── Slice colors ──────────────────────────────────────────────────────
        for i in range(len(values)):
            color = HexColor(palette[i % len(palette)])
            chart.slices[i].fillColor   = color
            chart.slices[i].strokeColor = HexColor(brand.colors.background)
            chart.slices[i].strokeWidth = 1

        chart.sideLabels         = True
        chart.sideLabelsOffset   = 0.1
        chart.slices.label_fontName  = brand.fonts.body
        chart.slices.label_fontSize  = brand.fonts.caption_size - 1
        chart.slices.label_fillColor = HexColor(brand.colors.text_dark)

        drawing.add(chart)
        renderPDF.draw(drawing, canvas, x, y)

    # ── Shared helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _format_value(val: float) -> str:
        """
        Format axis tick value compactly.
        1,500,000 → 1.5M
        12,500    → 12.5K
        150       → 150
        """
        abs_val = abs(val)
        if abs_val >= 1_000_000:
            return f"{val / 1_000_000:.1f}M"
        if abs_val >= 1_000:
            return f"{val / 1_000:.1f}K"
        return f"{val:.0f}"

    @staticmethod
    def _add_legend(
        drawing,
        series_names: list,
        palette:      list,
        brand,
        width:        float,
        y_pos:        float,
    ):
        """
        Draw a simple horizontal legend at the top of the drawing.
        """
        from reportlab.graphics.shapes import Rect, String
        from reportlab.lib.colors import HexColor

        box_size = 8
        gap      = 6
        text_gap = 4
        x        = 40

        for i, name in enumerate(series_names):
            color = HexColor(palette[i % len(palette)])

            # Color swatch
            swatch = Rect(x, y_pos - box_size, box_size, box_size)
            swatch.fillColor   = color
            swatch.strokeColor = color
            drawing.add(swatch)

            # Series name
            label = String(
                x + box_size + text_gap,
                y_pos - box_size + 1,
                name,
                fontSize  = brand.fonts.caption_size - 1,
                fontName  = brand.fonts.body,
                fillColor = HexColor(brand.colors.text_dark),
            )
            drawing.add(label)

            # Advance x for next legend item
            from reportlab.pdfbase.pdfmetrics import stringWidth
            x += (
                box_size
                + text_gap
                + stringWidth(name, brand.fonts.body, brand.fonts.caption_size - 1)
                + gap * 3
            )

################################################################################
# SECTION: PdfReportBuilder
################################################################################

class PdfReportBuilder:
    """
    Orchestrates PDF report generation.

    Responsibilities:
      - Load and register all section renderers
      - Create reportlab Canvas
      - Iterate sections → dispatch to renderer
      - Manage page breaks, headers, footers
      - Save to filepath

    Public API — one method:
        build(sections, brand, filepath) -> int  (page count)
    """

    def __init__(self):
        # ── Register all section renderers ────────────────────────────────────
        renderers = [
            PdfCoverSectionRenderer(),
            PdfHeadingSectionRenderer(),
            PdfTextSectionRenderer(),
            PdfBulletsSectionRenderer(),
            PdfTableSectionRenderer(),
            PdfMetricRowSectionRenderer(),
            PdfChartSectionRenderer(),
            PdfPageBreakSectionRenderer(),
        ]
        self._renderers: dict[str, AbstractPdfSectionRenderer] = {
            r.section_type: r for r in renderers
        }

    # ── Public API ────────────────────────────────────────────────────────────

    def build(
        self,
        sections: list,
        brand,              # PdfBrandGuidelines
        filepath: str,
        title:    str = "",
        author:   str = "",
    ) -> int:
        """
        Build PDF and save to filepath.
        Returns total page count.
        """
        # ── Page dimensions ───────────────────────────────────────────────────
        page_size              = brand.page_dimensions_pts()
        page_w, page_h         = page_size
        mt, mb, ml, mr         = brand.margin_pts()

        # ── Canvas ────────────────────────────────────────────────────────────
        canvas = Canvas(filepath, pagesize=page_size)
        canvas.setTitle(title or "Report")
        canvas.setAuthor(author or "")

        # ── Initial draw state ────────────────────────────────────────────────
        state = PdfDrawState(
            y             = page_h - mt,
            page_num      = 1,
            page_width    = page_w,
            page_height   = page_h,
            margin_top    = mt,
            margin_bottom = mb,
            margin_left   = ml,
            margin_right  = mr,
        )

        # ── Draw first page decorations ───────────────────────────────────────
        # Cover renderer handles its own page — skip decorations for page 1
        # if first section is a cover. Otherwise draw header/footer now.
        first_is_cover = (
            sections
            and sections[0].get("section_type") == "cover"
        )
        if not first_is_cover:
            self._draw_page_decorations(canvas, brand, state)

        # ── Render sections ───────────────────────────────────────────────────
        for section in sections:
            section_type = section.get("section_type", "")
            renderer     = self._renderers.get(section_type)

            if renderer is None:
                logger.warning(
                    "PdfReportBuilder: unknown section_type '%s' — skipped",
                    section_type,
                )
                continue

            renderer.render(
                canvas     = canvas,
                section    = section,
                brand      = brand,
                state      = state,
                on_new_page = self._on_new_page,
            )

        # ── Finalise last page ────────────────────────────────────────────────
        canvas.save()

        return state.page_num

    # ── Page management ───────────────────────────────────────────────────────

    def _on_new_page(
        self,
        canvas,
        brand,
        state: PdfDrawState,
    ):
        """
        Called by any renderer that needs a page break.
        Finalises current page, starts new page, draws decorations.
        Passed as a callable to each renderer — renderers never
        call canvas.showPage() directly.
        """
        canvas.showPage()
        state.page_num += 1
        state.reset_to_top()
        self._draw_page_decorations(canvas, brand, state)

    # ── Page decorations ──────────────────────────────────────────────────────

    def _draw_page_decorations(
        self,
        canvas,
        brand,
        state: PdfDrawState,
    ):
        """
        Draw header line, footer brand name, and page number.
        Called once per non-cover page.
        """
        W  = state.page_width
        H  = state.page_height
        ml = state.margin_left
        mr = state.margin_right
        mb = state.margin_bottom

        # ── Header line ───────────────────────────────────────────────────────
        if brand.rules.show_header_line:
            canvas.setStrokeColor(HexColor(brand.colors.primary))
            canvas.setLineWidth(1.5)
            canvas.line(
                ml,
                H - mb * 0.6,
                W - mr,
                H - mb * 0.6,
            )

            # Accent dot at left end of header line
            canvas.setFillColor(HexColor(brand.colors.accent))
            canvas.circle(ml, H - mb * 0.6, 3, stroke=0, fill=1)

        # ── Footer brand name ─────────────────────────────────────────────────
        if brand.rules.show_footer_brand:
            canvas.setFillColor(HexColor(brand.colors.text_muted))
            canvas.setFont(brand.fonts.body, brand.fonts.footer_size)
            canvas.drawString(
                ml,
                mb * 0.4,
                brand.brand_name,
            )

        # ── Footer separator line ─────────────────────────────────────────────
        canvas.setStrokeColor(HexColor(brand.colors.text_muted))
        canvas.setLineWidth(0.3)
        canvas.line(
            ml,
            mb * 0.6,
            W - mr,
            mb * 0.6,
        )

        # ── Page number ───────────────────────────────────────────────────────
        if brand.rules.show_page_numbers:
            canvas.setFillColor(HexColor(brand.colors.text_muted))
            canvas.setFont(brand.fonts.body, brand.fonts.footer_size)
            canvas.drawRightString(
                W - mr,
                mb * 0.4,
                f"Page {state.page_num}",
            )

        # ── Accent bar at bottom ──────────────────────────────────────────────
        bar_h = brand.rules.accent_bar_height * inch
        canvas.setFillColor(HexColor(brand.colors.accent))
        canvas.rect(0, 0, W, bar_h, stroke=0, fill=1)

# cmn/tools/tool/bedrock_converse_tools_pdf.py
# (append below the PdfBrandGuidelines section from Group 2)
