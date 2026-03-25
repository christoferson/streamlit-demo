import json
import logging
import os
import re
import tempfile
from dataclasses import dataclass, field
from typing import Any

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt, Emu

from cmn.tools.tool.bedrock_converse_tools_tool import AbstractBedrockConverseTool

logger = logging.getLogger(__name__)

# ── Brand config dir — relative to this file ─────────────────────────────────
_CONFIG_DIR = os.path.join(
    os.path.dirname(__file__),
    "..",
    "config",
    "pptx",
)

# ── Title layout constants ────────────────────────────────────────────────────
_TITLE_TOP_FRAC   = 0.05    # title top as fraction of slide height
_TITLE_HEIGHT_IN  = 0.55    # title text box height in inches
_TITLE_BAR_GAP_IN = 0.08    # gap between title text bottom and bar
_TITLE_BAR_H_IN   = 0.03    # bar thickness in inches


################################################################################
# SECTION: BrandGuidelines dataclasses
################################################################################

@dataclass
class BrandColors:
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

    def hex_to_rgb(self, hex_color: str) -> tuple:
        h = hex_color.lstrip("#")
        return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

    def get(self, name: str) -> str:
        """Look up color by attribute name — e.g. colors.get('primary')."""
        return getattr(self, name, self.primary)


@dataclass
class BrandFonts:
    heading:         str = "Calibri"
    body:            str = "Calibri"
    mono:            str = "Courier New"
    heading_size:    int = 36
    subheading_size: int = 24
    body_size:       int = 18
    caption_size:    int = 14


@dataclass
class BrandSlide:
    width_inches:  float = 13.33
    height_inches: float = 7.5


@dataclass
class BrandRules:
    accent_bar_height_inches: float = 0.08
    accent_bar_position:      str   = "bottom"
    logo_position:            str   = "bottom_right"
    max_bullets_per_slide:    int   = 6
    max_words_per_bullet:     int   = 15
    cover_background:         str   = "primary"
    section_background:       str   = "secondary"
    content_background:       str   = "background"
    use_slide_numbers:        bool  = True
    tone:                     str   = "professional"


@dataclass
class BrandLayoutConfig:
    title_y:       float = 0.05
    subtitle_y:    float = 0.55
    body_y:        float = 0.20
    quote_y:       float = 0.30
    attr_y:        float = 0.65
    split:         float = 0.50
    title_height:  float = 0.10
    title_bar_gap: float = 0.02


@dataclass
class BrandGuidelines:
    brand_name: str         = "Default"
    colors:     BrandColors = field(default_factory=BrandColors)
    fonts:      BrandFonts  = field(default_factory=BrandFonts)
    slide:      BrandSlide  = field(default_factory=BrandSlide)
    rules:      BrandRules  = field(default_factory=BrandRules)
    layouts:    dict        = field(default_factory=dict)

    # ── Factories ─────────────────────────────────────────────────────────────

    @classmethod
    def from_json(cls, path: str) -> "BrandGuidelines":
        with open(path, "r") as f:
            data = json.load(f)

        def _filter(dc, raw: dict) -> dict:
            return {
                k: v for k, v in raw.items()
                if k in dc.__dataclass_fields__
            }

        return cls(
            brand_name = data.get("brand_name", "Default"),
            colors     = BrandColors(**_filter(BrandColors, data.get("colors",  {}))),
            fonts      = BrandFonts( **_filter(BrandFonts,  data.get("fonts",   {}))),
            slide      = BrandSlide( **_filter(BrandSlide,  data.get("slide",   {}))),
            rules      = BrandRules( **_filter(BrandRules,  data.get("rules",   {}))),
            layouts    = data.get("layouts", {}),
        )

    @classmethod
    def load_all(cls, config_dir: str) -> dict:
        """
        Load every .json in config_dir.
        Returns { brand_name_lower: BrandGuidelines }
        Falls back to { "default": BrandGuidelines() } if dir missing.
        """
        brands = {}

        if not os.path.isdir(config_dir):
            logger.warning("Brand config dir not found: %s — using defaults", config_dir)
            return {"default": cls()}

        for fname in sorted(os.listdir(config_dir)):
            if not fname.endswith(".json"):
                continue
            path = os.path.join(config_dir, fname)
            try:
                brand = cls.from_json(path)
                brands[brand.brand_name.lower()] = brand
                logger.info("Loaded brand '%s' from %s", brand.brand_name, fname)
            except Exception as exc:
                logger.warning("Failed to load brand file %s: %s", fname, exc)

        if not brands:
            logger.warning("No brand files loaded — using built-in default")
            brands["default"] = cls()

        return brands

    # ── Helper ────────────────────────────────────────────────────────────────

    def layout(self, slide_type: str) -> BrandLayoutConfig:
        """Return BrandLayoutConfig for slide_type, with safe defaults."""
        raw   = self.layouts.get(slide_type, {})
        valid = {
            k: v for k, v in raw.items()
            if k in BrandLayoutConfig.__dataclass_fields__
        }
        return BrandLayoutConfig(**valid)


################################################################################
# SECTION: PptxBedrockConverseTool
################################################################################

class PptxBedrockConverseTool(AbstractBedrockConverseTool):

    def __init__(self, config_dir: str = _CONFIG_DIR):
        name = "create_pptx"
        definition = {
            "toolSpec": {
                "name": name,
                "description": (
                    "Creates a branded PowerPoint presentation (.pptx) and saves it "
                    "to a temporary file. Returns the file path for download. "
                    "Supported slide types: cover, section, content, "
                    "two_column, quote, closing. "
                    "Brand guidelines (colors, fonts, layout) are applied automatically "
                    "— do not specify colors or fonts in the slide definitions."
                ),
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "title": {
                                "type":        "string",
                                "description": (
                                    "Presentation title. Used in document properties "
                                    "and as the base for the saved filename."
                                ),
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
                            "slides": {
                                "type":        "array",
                                "description": "Ordered list of slide definitions.",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "slide_type": {
                                            "type": "string",
                                            "enum": [
                                                "cover", "section", "content",
                                                "two_column", "quote", "closing",
                                            ],
                                            "description": (
                                                "Layout type for this slide. Choose carefully:\n"
                                                "  cover      — title slide only. First slide of the deck.\n"
                                                "  section    — visual divider between major topics. "
                                                               "Title only, no body content. "
                                                               "Use ONLY as a separator, not for slides with content.\n"
                                                "  content    — ANY slide with a title + bullet points. "
                                                               "Use this for summaries, analysis, highlights, recommendations.\n"
                                                "  two_column — title + two side-by-side bullet columns. "
                                                               "Use for comparisons or before/after.\n"
                                                "  quote      — single large pull quote with attribution.\n"
                                                "  closing    — final thank you / contact slide.\n"
                                                "When in doubt, use 'content'."
                                            ),
                                        },
                                        "title": {
                                            "type":        "string",
                                            "description": "Slide title. Keep under 8 words.",
                                        },
                                        "subtitle": {
                                            "type":        "string",
                                            "description": (
                                                "Subtitle text. Used on cover and closing slides only. "
                                                "Do not use on content or section slides."
                                            ),
                                        },
                                        "section_number": {
                                            "type":        "integer",
                                            "description": (
                                                "Optional section number shown as a small label "
                                                "above the title on section slides. "
                                                "e.g. 1 renders as 'SECTION 01'. "
                                                "Increment by 1 for each section slide in the deck."
                                            ),
                                        },
                                        "bullets": {
                                            "type":        "array",
                                            "items":       {"type": "string"},
                                            "maxItems":    6,
                                            "description": (
                                                "Bullet points for this slide. "
                                                "Maximum 6 bullets. "
                                                "Keep each bullet under 15 words. "
                                                "If you have more than 6 points, "
                                                "split across multiple slides."
                                            ),
                                        },
                                        "left_header": {
                                            "type":        "string",
                                            "description": (
                                                "Left column heading. "
                                                "Used on two_column slides only."
                                            ),
                                        },
                                        "left_bullets": {
                                            "type":        "array",
                                            "items":       {"type": "string"},
                                            "maxItems":    6,
                                            "description": (
                                                "Left column bullets. "
                                                "Maximum 6. Keep each under 15 words."
                                            ),
                                        },
                                        "right_header": {
                                            "type":        "string",
                                            "description": (
                                                "Right column heading. "
                                                "Used on two_column slides only."
                                            ),
                                        },
                                        "right_bullets": {
                                            "type":        "array",
                                            "items":       {"type": "string"},
                                            "maxItems":    6,
                                            "description": (
                                                "Right column bullets. "
                                                "Maximum 6. Keep each under 15 words."
                                            ),
                                        },
                                        "quote": {
                                            "type":        "string",
                                            "description": (
                                                "Pull quote text. "
                                                "Used on quote slides only. One sentence."
                                            ),
                                        },
                                        "attribution": {
                                            "type":        "string",
                                            "description": (
                                                "Quote attribution. "
                                                "Used on quote slides only. "
                                                "e.g. '— VP Sales'"
                                            ),
                                        },
                                        "speaker_notes": {
                                            "type":        "string",
                                            "description": (
                                                "Speaker notes for presenter view. "
                                                "2-3 sentences of additional context "
                                                "not shown on the slide."
                                            ),
                                        },
                                    },
                                    "required": ["slide_type"],
                                },
                            },
                        },
                        "required": ["slides"],
                    }
                },
            }
        }
        super().__init__(name, definition)
        self._brands = BrandGuidelines.load_all(config_dir)

    # ── Summary ───────────────────────────────────────────────────────────────

    def summary(self) -> str:
        brands = ", ".join(sorted(self._brands.keys()))
        return (
            "create_pptx : creates a branded PowerPoint (.pptx) saved to a temp file. "
            "Provide slides list with slide_type, title, bullets, speaker_notes. "
            f"Available brands: {brands}."
        )

    # ── Filename helper ───────────────────────────────────────────────────────

    @staticmethod
    def _safe_filename(title: str) -> str:
        """Safe cross-platform filename — strips chars invalid on Windows/macOS/URLs."""
        safe = re.sub(r'[<>:"/\\|?*]', '-', title)
        safe = re.sub(r'[-\s]+', '_', safe).strip('_')
        return safe[:100] or "presentation"

    # ── Invoke ────────────────────────────────────────────────────────────────

    def invoke(self, params, tool_args: dict = None) -> dict:
        args      = tool_args or {}
        slides    = args.get("slides",  [])
        title     = args.get("title",   "Presentation")
        author    = args.get("author",  "")
        brand_key = args.get("brand",   "default").lower()

        if not slides:
            return {
                "error": (
                    "No slides provided. "
                    "Pass a list of slide definitions with slide_type, title, bullets."
                )
            }

        warnings = []
        brand    = self._brands.get(brand_key)

        if brand is None:
            warnings.append(
                f"Brand '{brand_key}' not found. "
                f"Available: {list(self._brands.keys())}. Using default."
            )
            brand = self._brands.get("default", BrandGuidelines())

        try:
            prs = self._build_presentation(brand, slides, title, author, warnings)
        except Exception as exc:
            logger.exception("PptxTool: build failed")
            return {"error": f"Failed to build presentation: {exc}"}

        safe_title = self._safe_filename(title)
        filename   = f"{safe_title}.pptx"

        tmp = tempfile.NamedTemporaryFile(
            suffix  = ".pptx",
            prefix  = f"pptx_{safe_title}_",
            delete  = False,
        )
        prs.save(tmp.name)
        tmp.close()

        logger.info("PptxTool: saved '%s' → %s", filename, tmp.name)

        return {
            "status":       "pptx_ready",
            "title":        title,
            "brand":        brand.brand_name,
            "slide_count":  len(slides),
            "slide_titles": [s.get("title", "") for s in slides],
            "filepath":     tmp.name,
            "filename":     filename,
            "warnings":     warnings,
        }

    # ── Presentation Builder ──────────────────────────────────────────────────

    def _build_presentation(
        self,
        brand:    BrandGuidelines,
        slides:   list,
        title:    str,
        author:   str,
        warnings: list,
    ) -> Presentation:

        prs              = Presentation()
        prs.slide_width  = Inches(brand.slide.width_inches)
        prs.slide_height = Inches(brand.slide.height_inches)

        core        = prs.core_properties
        core.title  = title
        core.author = author

        blank_layout = prs.slide_layouts[6]

        for idx, slide_def in enumerate(slides):
            slide_type = slide_def.get("slide_type", "content")
            slide      = prs.slides.add_slide(blank_layout)

            builder = self._BUILDERS.get(slide_type, self._build_content)
            builder(self, slide, slide_def, brand, warnings)

            self._add_accent_bar(slide, brand)

            if brand.rules.use_slide_numbers:
                self._add_slide_number(slide, brand, idx + 1)

            notes = slide_def.get("speaker_notes", "")
            if notes:
                slide.notes_slide.notes_text_frame.text = notes

        return prs

    # ── Slide Builders ────────────────────────────────────────────────────────

    def _build_cover(self, slide, slide_def, brand, warnings):
        W  = brand.slide.width_inches
        H  = brand.slide.height_inches
        lc = brand.layout("cover")

        self._fill_background(
            slide,
            brand.colors.hex_to_rgb(brand.colors.get(brand.rules.cover_background))
        )

        if title := slide_def.get("title", ""):
            self._add_text_box(
                slide,
                text      = title,
                left      = Inches(0.8),
                top       = Inches(H * lc.title_y),
                width     = Inches(W - 1.6),
                height    = Inches(1.2),
                font_name = brand.fonts.heading,
                font_size = brand.fonts.heading_size + 8,
                bold      = True,
                color_rgb = brand.colors.hex_to_rgb(brand.colors.text_light),
                align     = PP_ALIGN.LEFT,
            )

        if subtitle := slide_def.get("subtitle", ""):
            self._add_text_box(
                slide,
                text      = subtitle,
                left      = Inches(0.8),
                top       = Inches(H * lc.subtitle_y),
                width     = Inches(W - 1.6),
                height    = Inches(0.8),
                font_name = brand.fonts.body,
                font_size = brand.fonts.subheading_size,
                bold      = False,
                color_rgb = brand.colors.hex_to_rgb(brand.colors.accent),
                align     = PP_ALIGN.LEFT,
            )

    def _build_section(self, slide, slide_def, brand, warnings):
        W  = brand.slide.width_inches
        H  = brand.slide.height_inches

        # ── Background: light surface — clearly different from cover ──────────
        self._fill_background(
            slide,
            brand.colors.hex_to_rgb(brand.colors.surface)
        )

        # ── Left accent stripe ────────────────────────────────────────────────
        stripe_w = 0.18
        stripe_h = H * 0.50
        stripe_t = H * 0.25

        stripe = slide.shapes.add_shape(
            1,
            Inches(0.35),
            Inches(stripe_t),
            Inches(stripe_w),
            Inches(stripe_h),
        )
        stripe.fill.solid()
        stripe.fill.fore_color.rgb = RGBColor(
            *brand.colors.hex_to_rgb(brand.colors.accent)
        )
        stripe.line.fill.background()

        # ── Content left margin — starts after stripe + gap ───────────────────
        content_left = 0.35 + stripe_w + 0.25

        # ── Section label — "SECTION 01" ──────────────────────────────────────
        section_number = slide_def.get("section_number")
        label_top_in   = H * 0.28

        label = (
            f"SECTION {int(section_number):02d}"
            if section_number is not None
            else "SECTION"
        )

        self._add_text_box(
            slide,
            text      = label,
            left      = Inches(content_left),
            top       = Inches(label_top_in),
            width     = Inches(W - content_left - 0.5),
            height    = Inches(0.3),
            font_name = brand.fonts.body,
            font_size = brand.fonts.caption_size,
            bold      = False,
            color_rgb = brand.colors.hex_to_rgb(brand.colors.text_muted),
            align     = PP_ALIGN.LEFT,
        )

        # ── Title ─────────────────────────────────────────────────────────────
        title_top_in    = label_top_in + 0.35
        title_height_in = 1.1

        if title := slide_def.get("title", ""):
            self._add_text_box(
                slide,
                text      = title,
                left      = Inches(content_left),
                top       = Inches(title_top_in),
                width     = Inches(W - content_left - 0.5),
                height    = Inches(title_height_in),
                font_name = brand.fonts.heading,
                font_size = brand.fonts.heading_size + 4,
                bold      = True,
                color_rgb = brand.colors.hex_to_rgb(brand.colors.primary),
                align     = PP_ALIGN.LEFT,
            )

        # ── Subtitle ──────────────────────────────────────────────────────────
        subtitle_top_in = title_top_in + title_height_in + 0.1

        if subtitle := slide_def.get("subtitle", ""):
            self._add_text_box(
                slide,
                text      = subtitle,
                left      = Inches(content_left),
                top       = Inches(subtitle_top_in),
                width     = Inches(W - content_left - 0.5),
                height    = Inches(0.5),
                font_name = brand.fonts.body,
                font_size = brand.fonts.subheading_size - 4,
                bold      = False,
                color_rgb = brand.colors.hex_to_rgb(brand.colors.accent),
                align     = PP_ALIGN.LEFT,
            )

    def _build_content(self, slide, slide_def, brand, warnings):
        W  = brand.slide.width_inches
        H  = brand.slide.height_inches

        self._fill_background(
            slide,
            brand.colors.hex_to_rgb(brand.colors.get(brand.rules.content_background))
        )

        title_top_in = H * _TITLE_TOP_FRAC
        bar_top_in   = self._title_bar_top(brand)
        body_top_in  = self._body_top(brand)

        self._add_title_bar(slide, brand, top_inches=bar_top_in)

        if title := slide_def.get("title", ""):
            self._add_text_box(
                slide,
                text      = title,
                left      = Inches(0.4),
                top       = Inches(title_top_in),
                width     = Inches(W - 0.8),
                height    = Inches(_TITLE_HEIGHT_IN),
                font_name = brand.fonts.heading,
                font_size = brand.fonts.subheading_size,
                bold      = True,
                color_rgb = brand.colors.hex_to_rgb(brand.colors.primary),
                align     = PP_ALIGN.LEFT,
            )

        bullets = slide_def.get("bullets", [])
        if len(bullets) > brand.rules.max_bullets_per_slide:
            warnings.append(
                f"Slide '{slide_def.get('title', '')}': "
                f"{len(bullets)} bullets exceeds recommended max "
                f"{brand.rules.max_bullets_per_slide}. "
                f"Consider splitting into two slides."
            )

        if bullets:
            self._add_bullet_box(
                slide,
                bullets = bullets,
                left    = Inches(0.5),
                top     = Inches(body_top_in),
                width   = Inches(W - 1.0),
                height  = Inches(H - body_top_in - 0.4),
                brand   = brand,
            )

    def _build_two_column(self, slide, slide_def, brand, warnings):
        W  = brand.slide.width_inches
        H  = brand.slide.height_inches
        lc = brand.layout("two_column")

        self._fill_background(
            slide,
            brand.colors.hex_to_rgb(brand.colors.get(brand.rules.content_background))
        )

        title_top_in = H * _TITLE_TOP_FRAC
        bar_top_in   = self._title_bar_top(brand)
        body_top_in  = self._body_top(brand)

        self._add_title_bar(slide, brand, top_inches=bar_top_in)

        if title := slide_def.get("title", ""):
            self._add_text_box(
                slide,
                text      = title,
                left      = Inches(0.4),
                top       = Inches(title_top_in),
                width     = Inches(W - 0.8),
                height    = Inches(_TITLE_HEIGHT_IN),
                font_name = brand.fonts.heading,
                font_size = brand.fonts.subheading_size,
                bold      = True,
                color_rgb = brand.colors.hex_to_rgb(brand.colors.primary),
                align     = PP_ALIGN.LEFT,
            )

        col_w     = (W - 1.2) * lc.split
        body_h_in = H - body_top_in - 0.4

        for side, x_left, header_key, bullets_key in [
            ("left",  Inches(0.5),         "left_header",  "left_bullets"),
            ("right", Inches(0.7 + col_w), "right_header", "right_bullets"),
        ]:
            if header := slide_def.get(header_key, ""):
                self._add_text_box(
                    slide,
                    text      = header,
                    left      = x_left,
                    top       = Inches(body_top_in),
                    width     = Inches(col_w),
                    height    = Inches(0.4),
                    font_name = brand.fonts.heading,
                    font_size = brand.fonts.body_size,
                    bold      = True,
                    color_rgb = brand.colors.hex_to_rgb(brand.colors.secondary),
                    align     = PP_ALIGN.LEFT,
                )

            if bullets := slide_def.get(bullets_key, []):
                if len(bullets) > brand.rules.max_bullets_per_slide:
                    warnings.append(
                        f"Slide '{slide_def.get('title', '')}' {side} column: "
                        f"{len(bullets)} bullets exceeds recommended max "
                        f"{brand.rules.max_bullets_per_slide}."
                    )
                self._add_bullet_box(
                    slide,
                    bullets = bullets,
                    left    = x_left,
                    top     = Inches(body_top_in + 0.45),
                    width   = Inches(col_w),
                    height  = Inches(body_h_in),
                    brand   = brand,
                )

        self._add_vertical_divider(slide, brand, col_w, body_top_in / H, H)

    def _build_quote(self, slide, slide_def, brand, warnings):
        W  = brand.slide.width_inches
        H  = brand.slide.height_inches
        lc = brand.layout("quote")

        self._fill_background(
            slide,
            brand.colors.hex_to_rgb(brand.colors.surface)
        )

        if quote := slide_def.get("quote", ""):
            self._add_text_box(
                slide,
                text      = f"\u201c{quote}\u201d",
                left      = Inches(1.2),
                top       = Inches(H * lc.quote_y),
                width     = Inches(W - 2.4),
                height    = Inches(2.0),
                font_name = brand.fonts.heading,
                font_size = brand.fonts.heading_size - 4,
                bold      = False,
                italic    = True,
                color_rgb = brand.colors.hex_to_rgb(brand.colors.primary),
                align     = PP_ALIGN.CENTER,
            )

        if attribution := slide_def.get("attribution", ""):
            self._add_text_box(
                slide,
                text      = attribution,
                left      = Inches(1.2),
                top       = Inches(H * lc.attr_y),
                width     = Inches(W - 2.4),
                height    = Inches(0.4),
                font_name = brand.fonts.body,
                font_size = brand.fonts.caption_size,
                bold      = False,
                color_rgb = brand.colors.hex_to_rgb(brand.colors.text_muted),
                align     = PP_ALIGN.RIGHT,
            )

    def _build_closing(self, slide, slide_def, brand, warnings):
        """Closing reuses cover layout."""
        self._build_cover(slide, slide_def, brand, warnings)

    # ── Dispatch table ────────────────────────────────────────────────────────

    _BUILDERS = {
        "cover":      _build_cover,
        "section":    _build_section,
        "content":    _build_content,
        "two_column": _build_two_column,
        "quote":      _build_quote,
        "closing":    _build_closing,
    }

    # ── Drawing Helpers ───────────────────────────────────────────────────────

    def _title_bar_top(self, brand: BrandGuidelines) -> float:
        """Bar top in inches: title_top + title_height + small gap above bar."""
        H = brand.slide.height_inches
        return (H * _TITLE_TOP_FRAC) + _TITLE_HEIGHT_IN + _TITLE_BAR_GAP_IN

    def _body_top(self, brand: BrandGuidelines) -> float:
        """Body starts immediately after bar — no extra gap below."""
        return self._title_bar_top(brand) + _TITLE_BAR_H_IN + 0.10

    def _fill_background(self, slide, rgb: tuple):
        fill = slide.background.fill
        fill.solid()
        fill.fore_color.rgb = RGBColor(*rgb)

    def _add_title_bar(self, slide, brand: BrandGuidelines, top_inches: float):
        W     = brand.slide.width_inches
        shape = slide.shapes.add_shape(
            1,
            Inches(0.4),
            Inches(top_inches),
            Inches(W - 0.8),
            Inches(_TITLE_BAR_H_IN),
        )
        shape.fill.solid()
        shape.fill.fore_color.rgb = RGBColor(
            *brand.colors.hex_to_rgb(brand.colors.primary)
        )
        shape.line.fill.background()

    def _add_accent_bar(self, slide, brand: BrandGuidelines):
        """Full-width accent bar at bottom (or top) of every slide."""
        W     = brand.slide.width_inches
        H     = brand.slide.height_inches
        bar_h = brand.rules.accent_bar_height_inches
        top   = (
            Inches(H - bar_h)
            if brand.rules.accent_bar_position == "bottom"
            else Inches(0)
        )
        shape = slide.shapes.add_shape(
            1, Inches(0), top, Inches(W), Inches(bar_h),
        )
        shape.fill.solid()
        shape.fill.fore_color.rgb = RGBColor(
            *brand.colors.hex_to_rgb(brand.colors.accent)
        )
        shape.line.fill.background()

    def _add_slide_number(self, slide, brand: BrandGuidelines, number: int):
        W = brand.slide.width_inches
        H = brand.slide.height_inches
        self._add_text_box(
            slide,
            text      = str(number),
            left      = Inches(W - 0.5),
            top       = Inches(H - 0.35),
            width     = Inches(0.4),
            height    = Inches(0.25),
            font_name = brand.fonts.body,
            font_size = brand.fonts.caption_size - 2,
            bold      = False,
            color_rgb = brand.colors.hex_to_rgb(brand.colors.text_muted),
            align     = PP_ALIGN.RIGHT,
        )

    def _add_vertical_divider(self, slide, brand, col_w, body_y, H):
        shape = slide.shapes.add_shape(
            1,
            Inches(0.6 + col_w), Inches(H * body_y),
            Inches(0.02),        Inches(H - body_y - 0.4),
        )
        shape.fill.solid()
        shape.fill.fore_color.rgb = RGBColor(
            *brand.colors.hex_to_rgb(brand.colors.secondary)
        )
        shape.line.fill.background()

    def _add_text_box(
        self,
        slide,
        text:      str,
        left:      Emu,
        top:       Emu,
        width:     Emu,
        height:    Emu,
        font_name: str,
        font_size: int,
        bold:      bool,
        color_rgb: tuple,
        align:     PP_ALIGN = PP_ALIGN.LEFT,
        italic:    bool = False,
    ):
        tf           = slide.shapes.add_textbox(left, top, width, height).text_frame
        tf.word_wrap = True
        p            = tf.paragraphs[0]
        p.alignment  = align
        run                = p.add_run()
        run.text           = text
        run.font.name      = font_name
        run.font.size      = Pt(font_size)
        run.font.bold      = bold
        run.font.italic    = italic
        run.font.color.rgb = RGBColor(*color_rgb)

    def _add_bullet_box(
        self,
        slide,
        bullets: list,
        left:    Emu,
        top:     Emu,
        width:   Emu,
        height:  Emu,
        brand:   BrandGuidelines,
    ):
        tf           = slide.shapes.add_textbox(left, top, width, height).text_frame
        tf.word_wrap = True
        text_rgb     = brand.colors.hex_to_rgb(brand.colors.text_dark)
        accent_rgb   = brand.colors.hex_to_rgb(brand.colors.accent)

        for i, bullet in enumerate(bullets):
            p              = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.alignment    = PP_ALIGN.LEFT
            p.space_before = Pt(4)

            marker                = p.add_run()
            marker.text           = "▪  "
            marker.font.name      = brand.fonts.body
            marker.font.size      = Pt(brand.fonts.body_size)
            marker.font.color.rgb = RGBColor(*accent_rgb)

            run                = p.add_run()
            run.text           = bullet
            run.font.name      = brand.fonts.body
            run.font.size      = Pt(brand.fonts.body_size)
            run.font.color.rgb = RGBColor(*text_rgb)