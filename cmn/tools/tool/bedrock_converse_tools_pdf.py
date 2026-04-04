# cmn/tools/tool/bedrock_converse_tools_pdf.py

import json
import logging
import os
import re
import tempfile
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

_CONFIG_DIR = os.path.join(
    os.path.dirname(__file__),
    "..", "config", "pdf",
)

################################################################################
# SECTION: PdfBrandGuidelines
################################################################################

@dataclass
class PdfBrandColors:
    primary:       str = "#1B3A6B"
    secondary:     str = "#2E86AB"
    accent:        str = "#F4A261"
    background:    str = "#FFFFFF"
    surface:       str = "#F5F7FA"
    text_dark:     str = "#1A1A2E"
    text_light:    str = "#FFFFFF"
    text_muted:    str = "#6B7280"
    success:       str = "#10B981"
    warning:       str = "#F59E0B"
    danger:        str = "#EF4444"
    chart_palette: list = field(default_factory=lambda: [
        "#1B3A6B", "#2E86AB", "#F4A261",
        "#10B981", "#F59E0B", "#EF4444",
    ])

    def hex_to_rgb_float(self, hex_color: str) -> tuple:
        """
        Returns (r, g, b) as 0.0–1.0 floats.
        reportlab Color() expects floats not ints.
        """
        h = hex_color.lstrip("#")
        r, g, b = (int(h[i:i+2], 16) for i in (0, 2, 4))
        return r / 255, g / 255, b / 255

    def get(self, name: str) -> str:
        """Look up color by attribute name — e.g. colors.get('primary')."""
        return getattr(self, name, self.primary)

    def to_rl_color(self, hex_color: str):
        """Convert hex string → reportlab Color object."""
        from reportlab.lib.colors import Color
        r, g, b = self.hex_to_rgb_float(hex_color)
        return Color(r, g, b)


@dataclass
class PdfBrandFonts:
    heading:          str = "Helvetica-Bold"
    body:             str = "Helvetica"
    mono:             str = "Courier"
    cover_title_size: int = 32
    h1_size:          int = 20
    h2_size:          int = 16
    body_size:        int = 11
    caption_size:     int = 9
    footer_size:      int = 8


@dataclass
class PdfBrandPage:
    size:          str   = "A4"
    margin_top:    float = 1.0
    margin_bottom: float = 1.0
    margin_left:   float = 1.0
    margin_right:  float = 1.0


@dataclass
class PdfBrandRules:
    show_page_numbers:    bool  = True
    show_footer_brand:    bool  = True
    show_header_line:     bool  = True
    accent_bar_height:    float = 0.08
    table_row_height:     float = 0.3
    section_space_before: float = 0.2
    bullet_indent:        float = 0.3
    chart_height_inches:  float = 3.5


@dataclass
class PdfBrandGuidelines:
    brand_name: str           = "Default"
    colors:     PdfBrandColors = field(default_factory=PdfBrandColors)
    fonts:      PdfBrandFonts  = field(default_factory=PdfBrandFonts)
    page:       PdfBrandPage   = field(default_factory=PdfBrandPage)
    rules:      PdfBrandRules  = field(default_factory=PdfBrandRules)

    # ── Factories ─────────────────────────────────────────────────────────────

    @classmethod
    def from_json(cls, path: str) -> "PdfBrandGuidelines":
        with open(path, "r") as f:
            data = json.load(f)

        def _filter(dc, raw: dict) -> dict:
            """Only pass keys that exist in the dataclass — future-proof."""
            return {
                k: v for k, v in raw.items()
                if k in dc.__dataclass_fields__
            }

        return cls(
            brand_name = data.get("brand_name", "Default"),
            colors     = PdfBrandColors(**_filter(PdfBrandColors, data.get("colors", {}))),
            fonts      = PdfBrandFonts( **_filter(PdfBrandFonts,  data.get("fonts",  {}))),
            page       = PdfBrandPage(  **_filter(PdfBrandPage,   data.get("page",   {}))),
            rules      = PdfBrandRules( **_filter(PdfBrandRules,  data.get("rules",  {}))),
        )

    @classmethod
    def load_all(cls, config_dir: str) -> dict:
        """
        Load every .json in config_dir.
        Returns { brand_name_lower: PdfBrandGuidelines }.
        Falls back to { "default": PdfBrandGuidelines() } if dir missing or empty.
        """
        brands = {}

        if not os.path.isdir(config_dir):
            logger.warning(
                "PDF brand config dir not found: %s — using built-in default",
                config_dir,
            )
            return {"default": cls()}

        for fname in sorted(os.listdir(config_dir)):
            if not fname.endswith(".json"):
                continue
            path = os.path.join(config_dir, fname)
            try:
                brand = cls.from_json(path)
                brands[brand.brand_name.lower()] = brand
                logger.info("Loaded PDF brand '%s' from %s", brand.brand_name, fname)
            except Exception as exc:
                logger.warning("Failed to load PDF brand file %s: %s", fname, exc)

        if not brands:
            logger.warning("No PDF brand files loaded — using built-in default")
            brands["default"] = cls()

        return brands

    # ── Helpers ───────────────────────────────────────────────────────────────

    def page_dimensions_pts(self) -> tuple:
        """
        Returns (width, height) in reportlab points.
        reportlab uses points (1 inch = 72 pts).
        """
        from reportlab.lib.pagesizes import A4, LETTER, A3
        size_map = {
            "A4":     A4,
            "LETTER": LETTER,
            "A3":     A3,
        }
        return size_map.get(self.page.size.upper(), A4)

    def margin_pts(self) -> tuple:
        """
        Returns (top, bottom, left, right) in points.
        """
        pts = 72   # 1 inch = 72 points
        return (
            self.page.margin_top    * pts,
            self.page.margin_bottom * pts,
            self.page.margin_left   * pts,
            self.page.margin_right  * pts,
        )


import re
import tempfile
from cmn.tools.tool.bedrock_converse_tools_tool import AbstractBedrockConverseTool
from cmn.tools.tool.pdf import PdfReportBuilder


################################################################################
# SECTION: Filename helper
################################################################################

def _safe_filename(title: str) -> str:
    """Convert title to safe cross-platform filename."""
    safe = re.sub(r'[<>:"/\\|?*]', "-", title)
    safe = re.sub(r"[-\s]+", "_", safe).strip("_")
    return safe[:100] or "report"


################################################################################
# SECTION: PdfBedrockConverseTool
################################################################################

class PdfBedrockConverseTool(AbstractBedrockConverseTool):

    def __init__(self, config_dir: str = _CONFIG_DIR):
        name = "create_pdf"
        definition = {
            "toolSpec": {
                "name": name,
                "description": (
                    "Creates a branded PDF report with text, tables, and charts. "
                    "The report is domain-agnostic — works for sales, inventory, "
                    "finance, HR, or any structured data. "
                    "For chart sections: fetch data from the relevant tool first "
                    "if not already in conversation context, then reshape into chart_data. "
                    "Brand guidelines (colors, fonts, layout) are applied automatically. "
                    "IMPORTANT: When the user asks for a report, document, or PDF "
                    "use this tool. When the user asks for a presentation or slides "
                    "use create_pptx instead. "
                    "Section types: cover, heading, text, bullets, "
                    "table, metric_row, chart, page_break."
                ),
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {

                            "title": {
                                "type":        "string",
                                "description": "Report title used in document properties and filename.",
                            },
                            "author": {
                                "type":        "string",
                                "description": "Author name for document properties.",
                            },
                            "brand": {
                                "type":        "string",
                                "description": (
                                    "Brand name to apply. "
                                    "Defaults to 'default' if omitted or not found."
                                ),
                                "default": "default",
                            },

                            "sections": {
                                "type":        "array",
                                "description": "Ordered list of report section definitions.",
                                "items": {
                                    "type": "object",
                                    "properties": {

                                        # ── Section type ──────────────────────
                                        "section_type": {
                                            "type": "string",
                                            "enum": [
                                                "cover",
                                                "heading",
                                                "text",
                                                "bullets",
                                                "table",
                                                "metric_row",
                                                "chart",
                                                "page_break",
                                            ],
                                            "description": (
                                                "cover: full-page title page with brand background. "
                                                "heading: section title, level 1 (large) or 2 (medium). "
                                                "text: body paragraph, wraps automatically. "
                                                "bullets: bullet list of items. "
                                                "table: data table with headers and rows. "
                                                "metric_row: KPI boxes in a horizontal row (max 4). "
                                                "chart: native vector chart — bar, line, or pie. "
                                                "  USE THIS when showing numeric trends or comparisons. "
                                                "  NEVER put time-series data into text or bullets. "
                                                "page_break: force content onto a new page."
                                            ),
                                        },

                                        # ── Cover fields ──────────────────────
                                        "title": {
                                            "type":        "string",
                                            "description": "Title text (cover and heading sections).",
                                        },
                                        "subtitle": {
                                            "type":        "string",
                                            "description": "Subtitle (cover section only).",
                                        },
                                        "author": {
                                            "type":        "string",
                                            "description": "Author name (cover section only).",
                                        },
                                        "date": {
                                            "type":        "string",
                                            "description": (
                                                "Date string (cover section only). "
                                                "Auto-populated with today's date if omitted."
                                            ),
                                        },

                                        # ── Heading fields ────────────────────
                                        "text": {
                                            "type":        "string",
                                            "description": (
                                                "Text content for heading and text sections. "
                                                "For heading: the heading title. "
                                                "For text: the paragraph body."
                                            ),
                                        },
                                        "level": {
                                            "type":        "integer",
                                            "enum":        [1, 2],
                                            "description": (
                                                "Heading level. "
                                                "1: large primary heading with underline rule. "
                                                "2: medium secondary heading, no rule."
                                            ),
                                        },

                                        # ── Bullets fields ────────────────────
                                        "items": {
                                            "type":        "array",
                                            "items":       {"type": "string"},
                                            "description": "Bullet list items (bullets section only).",
                                        },

                                        # ── Table fields ──────────────────────
                                        "headers": {
                                            "type":        "array",
                                            "items":       {"type": "string"},
                                            "description": "Column header labels (table section).",
                                        },
                                        "rows": {
                                            "type":  "array",
                                            "items": {
                                                "type":  "array",
                                                "items": {"type": "string"},
                                            },
                                            "description": (
                                                "Table data rows (table section). "
                                                "Each row is a list of strings matching headers order. "
                                                "Example: [[\"January\", \"$213,000\", \"1,035\"]]."
                                            ),
                                        },

                                        # ── Metric row fields ─────────────────
                                        "metrics": {
                                            "type": "array",
                                            "items": {
                                                "type": "object",
                                                "properties": {
                                                    "label": {
                                                        "type":        "string",
                                                        "description": "Metric label.",
                                                    },
                                                    "value": {
                                                        "type":        "string",
                                                        "description": "Metric value as formatted string.",
                                                    },
                                                    "delta": {
                                                        "type":        "string",
                                                        "description": (
                                                            "Change indicator. "
                                                            "Prefix with + for positive (green ▲), "
                                                            "- for negative (red ▼), "
                                                            "omit for neutral (grey ─)."
                                                        ),
                                                    },
                                                },
                                                "required": ["label", "value"],
                                            },
                                            "maxItems":    4,
                                            "description": (
                                                "KPI metrics for metric_row section. "
                                                "Maximum 4 per row. "
                                                "Use multiple metric_row sections for more than 4."
                                            ),
                                        },

                                        # ── Chart fields ──────────────────────
                                        "chart_type": {
                                            "type": "string",
                                            "enum": ["bar", "line", "pie"],
                                            "description": (
                                                "Chart type. Required for chart sections. "
                                                "bar: comparisons between categories. "
                                                "line: trends over time — use for monthly/time-series. "
                                                "pie: proportions (max 6 slices recommended)."
                                            ),
                                        },
                                        "chart_data": {
                                            "type": "array",
                                            "description": (
                                                "REQUIRED for chart sections. "
                                                "Fetch from relevant data tool first if not in context. "
                                                "Single series:  [{\"label\": \"Jan\", \"value\": 213000}]. "
                                                "Multi-series:   [{\"label\": \"Jan\", \"value\": 213000, \"series\": \"2024\"}]. "
                                                "label:  category (month, product, region, date, etc.). "
                                                "value:  numeric metric (revenue, units, price, etc.). "
                                                "series: group name for multi-series (year, region, etc.)."
                                            ),
                                            "items": {
                                                "type": "object",
                                                "properties": {
                                                    "label":  {
                                                        "type":        "string",
                                                        "description": "Category label.",
                                                    },
                                                    "value":  {
                                                        "type":        "number",
                                                        "description": "Numeric value.",
                                                    },
                                                    "series": {
                                                        "type":        "string",
                                                        "description": "Series name for multi-series.",
                                                    },
                                                },
                                                "required": ["label", "value"],
                                            },
                                        },
                                        "chart_title": {
                                            "type":        "string",
                                            "description": "Title shown above the chart.",
                                        },
                                        "x_label": {
                                            "type":        "string",
                                            "description": "X-axis label.",
                                        },
                                        "y_label": {
                                            "type":        "string",
                                            "description": "Y-axis label.",
                                        },

                                    },
                                    "required": ["section_type"],
                                },
                            },
                        },
                        "required": ["sections"],
                    }
                },
            }
        }
        super().__init__(name, definition)

        self._brands  = PdfBrandGuidelines.load_all(config_dir)
        self._builder = PdfReportBuilder()

    # ── Summary ───────────────────────────────────────────────────────────────

    def summary(self) -> str:
        brands = ", ".join(sorted(self._brands.keys()))
        return (
            "create_pdf : creates a branded PDF report with text, tables, "
            "and native vector charts. "

            # ── Section types ─────────────────────────────────────────────────
            "Section types: "
            "cover (full-page title), "
            "heading (level 1 or 2), "
            "text (paragraph), "
            "bullets (bullet list), "
            "table (data table with headers + rows), "
            "metric_row (KPI boxes, max 4 per row — "
            "use multiple metric_row sections for more), "
            "chart (native vector chart — "
            "USE THIS for numeric data, trends, comparisons), "
            "page_break (force new page). "

            # ── Hard rule ─────────────────────────────────────────────────────
            "RULE: When you have numeric data (revenue, units, prices, counts) "
            "always use section_type 'chart'. "
            "NEVER put time-series or comparative numeric data into text or bullets only. "

            # ── Chart types ───────────────────────────────────────────────────
            "Chart types: "
            "bar (comparisons), "
            "line (trends over time — use for monthly/time-series data), "
            "pie (proportions). "

            # ── Data flow ─────────────────────────────────────────────────────
            "For chart sections: "
            "if data is already in conversation context use it directly, "
            "otherwise fetch from the relevant data tool first. "
            "Reshape into chart_data: "
            "single series [{label, value}] or "
            "multi-series [{label, value, series}]. "

            # ── Recommended structure ─────────────────────────────────────────
            "RECOMMENDED structure for a data report: "
            "1. cover, "
            "2. heading (level 1) + metric_row (KPIs), "
            "3. heading (level 2) + chart (primary metric), "
            "4. heading (level 2) + chart (secondary metric), "
            "5. heading (level 2) + table (detailed data), "
            "6. heading (level 1) + bullets (insights/recommendations). "
            "ALWAYS include at least one chart section when numeric data is available. "

            # ── vs PPT ────────────────────────────────────────────────────────
            "Use create_pdf for reports and documents. "
            "Use create_pptx for presentations and slides. "

            f"Available brands: {brands}."
        )

    # ── Invoke ────────────────────────────────────────────────────────────────

    def invoke(self, params, tool_args: dict = None) -> dict:
        args      = tool_args or {}
        sections  = args.get("sections", [])
        title     = args.get("title",    "Report")
        author    = args.get("author",   "")
        brand_key = args.get("brand",    "default").lower()

        if not sections:
            return {
                "error": (
                    "No sections provided. "
                    "Pass a list of section definitions with section_type "
                    "and relevant content fields."
                )
            }

        # ── Resolve brand ─────────────────────────────────────────────────────
        warnings = []
        brand    = self._brands.get(brand_key)

        if brand is None:
            warnings.append(
                f"Brand '{brand_key}' not found. "
                f"Available: {list(self._brands.keys())}. Using default."
            )
            brand = self._brands.get("default", PdfBrandGuidelines())

        # ── Save to temp file ─────────────────────────────────────────────────
        safe_title = _safe_filename(title)
        filename   = f"{safe_title}.pdf"

        tmp = tempfile.NamedTemporaryFile(
            suffix  = ".pdf",
            prefix  = f"pdf_{safe_title}_",
            delete  = False,
        )
        tmp.close()

        # ── Build PDF ─────────────────────────────────────────────────────────
        try:
            page_count = self._builder.build(
                sections = sections,
                brand    = brand,
                filepath = tmp.name,
                title    = title,
                author   = author,
            )
        except Exception as exc:
            logger.exception("PdfTool: build failed")
            return {"error": f"Failed to build PDF: {exc}"}

        logger.info("PdfTool: saved '%s' → %s (%d pages)", filename, tmp.name, page_count)

        return {
            "status":     "pdf_ready",
            "title":      title,
            "brand":      brand.brand_name,
            "pages":      page_count,
            "filepath":   tmp.name,
            "filename":   filename,
            "warnings":   warnings,
        }

