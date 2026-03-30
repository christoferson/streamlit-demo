from abc import ABC, abstractmethod
from collections import defaultdict
import logging

from pptx.chart.data import ChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import (
    XL_CHART_TYPE,
    XL_LEGEND_POSITION,
)
from pptx.util import Pt, Emu

logger = logging.getLogger(__name__)


################################################################################
# SECTION: AbstractPptxChartBuilder
################################################################################

class AbstractPptxChartBuilder(ABC):
    """
    Base class for all chart sub-builders.
    Each sub-builder owns exactly two responsibilities:
      1. xl_chart_type  — which python-pptx chart type to use
      2. build_chart_data() — convert generic list → ChartData
    """

    @property
    @abstractmethod
    def xl_chart_type(self) -> XL_CHART_TYPE:
        """python-pptx chart type enum."""

    @abstractmethod
    def build_chart_data(self, data: list) -> ChartData:
        """
        Convert generic chart_data list → ChartData.

        data format:
          Single series: [{"label": str, "value": float}, ...]
          Multi series:  [{"label": str, "value": float, "series": str}, ...]
        """

    # ── Shared helper — used by bar and line builders ─────────────────────────

    @staticmethod
    def _is_multi_series(data: list) -> bool:
        return any("series" in row for row in data)

    @staticmethod
    def _build_single(data: list, series_name: str = "Value") -> ChartData:
        cd = ChartData()
        cd.categories = [str(row["label"]) for row in data]
        cd.add_series(series_name, [float(row["value"]) for row in data])
        return cd

    @staticmethod
    def _build_multi(data: list) -> ChartData:
        """
        Groups rows by series name.
        Preserves label order from input.
        Fills missing label/series combinations with 0.
        """
        label_order = []
        series_map  = defaultdict(dict)   # series_name → {label: value}

        for row in data:
            label  = str(row["label"])
            value  = float(row["value"])
            series = str(row.get("series", "Value"))

            if label not in label_order:
                label_order.append(label)
            series_map[series][label] = value

        cd = ChartData()
        cd.categories = label_order

        for series_name, label_values in series_map.items():
            values = [label_values.get(lbl, 0) for lbl in label_order]
            cd.add_series(series_name, values)

        return cd


################################################################################
# SECTION: PptxBarChartBuilder
################################################################################

class PptxBarChartBuilder(AbstractPptxChartBuilder):
    """Vertical column chart — comparisons, rankings."""

    @property
    def xl_chart_type(self) -> XL_CHART_TYPE:
        return XL_CHART_TYPE.COLUMN_CLUSTERED

    def build_chart_data(self, data: list) -> ChartData:
        if self._is_multi_series(data):
            return self._build_multi(data)
        return self._build_single(data)


################################################################################
# SECTION: PptxLineChartBuilder
################################################################################

class PptxLineChartBuilder(AbstractPptxChartBuilder):
    """Line chart with markers — trends over time."""

    @property
    def xl_chart_type(self) -> XL_CHART_TYPE:
        return XL_CHART_TYPE.LINE_MARKERS

    def build_chart_data(self, data: list) -> ChartData:
        if self._is_multi_series(data):
            return self._build_multi(data)
        return self._build_single(data)


################################################################################
# SECTION: PptxPieChartBuilder
################################################################################

class PptxPieChartBuilder(AbstractPptxChartBuilder):
    """Pie chart — proportions. Single series only."""

    @property
    def xl_chart_type(self) -> XL_CHART_TYPE:
        return XL_CHART_TYPE.PIE

    def build_chart_data(self, data: list) -> ChartData:
        # Pie is always single series — ignore series field if present
        return self._build_single(data, series_name="")

    def style_pie_slices(self, plot, palette: list, hex_to_rgb):
        """
        Color each pie slice individually from brand palette.
        Called by PptxChartBuilder after chart is placed.
        """
        series = plot.series[0]
        for i, point in enumerate(series.points):
            color_hex = palette[i % len(palette)]
            rgb       = RGBColor(*hex_to_rgb(color_hex))
            point.format.fill.solid()
            point.format.fill.fore_color.rgb = rgb


################################################################################
# SECTION: PptxChartBuilder  (orchestrator)
################################################################################

class PptxChartBuilder:
    """
    Orchestrates chart creation on a python-pptx slide.

    Public API — one method:
        add_chart(slide, slide_def, brand, warnings, left, top, width, height)

    Responsibilities:
        - Resolve chart_type string → sub-builder
        - Delegate ChartData construction to sub-builder
        - Place chart shape on slide
        - Apply brand colors, fonts, legend, axes
    """

    _BUILDERS: dict[str, AbstractPptxChartBuilder] = {
        "bar":  PptxBarChartBuilder(),
        "line": PptxLineChartBuilder(),
        "pie":  PptxPieChartBuilder(),
    }

    _DEFAULT_CHART_TYPE = "bar"

    # ── Public API ────────────────────────────────────────────────────────────

    def add_chart(
        self,
        slide,
        slide_def: dict,
        brand,                  # BrandGuidelines — not imported here, passed in
        warnings:  list,
        left:      Emu,
        top:       Emu,
        width:     Emu,
        height:    Emu,
    ) -> bool:
        """
        Build and embed a native chart on the slide.
        Returns True on success, False if chart was skipped.
        """
        chart_data_raw = slide_def.get("chart_data",  [])
        chart_type_str = slide_def.get("chart_type",  self._DEFAULT_CHART_TYPE)
        chart_title    = slide_def.get("chart_title", slide_def.get("title", ""))
        x_label        = slide_def.get("x_label",     "")
        y_label        = slide_def.get("y_label",     "")
        slide_title    = slide_def.get("title",       "")

        # ── Validate ──────────────────────────────────────────────────────────
        if not chart_data_raw:
            warnings.append(
                f"Slide '{slide_title}': chart_data is empty — chart skipped."
            )
            return False

        # ── Resolve sub-builder ───────────────────────────────────────────────
        builder = self._BUILDERS.get(chart_type_str)
        if builder is None:
            warnings.append(
                f"Slide '{slide_title}': unknown chart_type '{chart_type_str}'. "
                f"Available: {list(self._BUILDERS.keys())}. "
                f"Falling back to '{self._DEFAULT_CHART_TYPE}'."
            )
            builder = self._BUILDERS[self._DEFAULT_CHART_TYPE]

        # ── Build ChartData ───────────────────────────────────────────────────
        try:
            chart_data = builder.build_chart_data(chart_data_raw)
        except Exception as exc:
            warnings.append(
                f"Slide '{slide_title}': chart data error — {exc}. Chart skipped."
            )
            logger.exception("PptxChartBuilder: build_chart_data failed")
            return False

        # ── Place chart on slide ──────────────────────────────────────────────
        try:
            graphic_frame = slide.shapes.add_chart(
                builder.xl_chart_type,
                left, top, width, height,
                chart_data,
            )
        except Exception as exc:
            warnings.append(
                f"Slide '{slide_title}': failed to add chart — {exc}. Chart skipped."
            )
            logger.exception("PptxChartBuilder: add_chart failed")
            return False

        chart      = graphic_frame.chart
        has_series = AbstractPptxChartBuilder._is_multi_series(chart_data_raw)

        # ── Apply brand styling ───────────────────────────────────────────────
        self._apply_title(chart, chart_title, brand)
        self._apply_series_colors(chart, chart_type_str, builder, brand)
        self._apply_legend(chart, has_series, brand)
        self._apply_axes(chart, chart_type_str, x_label, y_label, brand)

        return True

    # ── Styling — title ───────────────────────────────────────────────────────

    def _apply_title(self, chart, chart_title: str, brand):
        if chart_title:
            chart.has_title = True
            tf = chart.chart_title.text_frame
            tf.text = chart_title
            if tf.paragraphs and tf.paragraphs[0].runs:
                self._set_font(
                    tf.paragraphs[0].runs[0].font,
                    brand.fonts.body,
                    brand.fonts.body_size,
                    bold      = True,
                    hex_color = brand.colors.text_dark,
                    brand     = brand,
                )
        else:
            chart.has_title = False

    # ── Styling — series colors ───────────────────────────────────────────────

    def _apply_series_colors(self, chart, chart_type_str: str, builder, brand):
        """
        Dispatch to pie-specific or generic series coloring.
        """
        palette = brand.colors.chart_palette
        plot    = chart.plots[0]

        if chart_type_str == "pie" and isinstance(builder, PptxPieChartBuilder):
            builder.style_pie_slices(plot, palette, brand.colors.hex_to_rgb)
            return

        for i, series in enumerate(plot.series):
            color_hex = palette[i % len(palette)]
            rgb       = RGBColor(*brand.colors.hex_to_rgb(color_hex))

            if chart_type_str == "bar":
                series.format.fill.solid()
                series.format.fill.fore_color.rgb = rgb

            elif chart_type_str == "line":
                series.format.line.color.rgb = rgb
                series.format.line.width     = Pt(2.0)
                marker = series.marker
                marker.format.fill.solid()
                marker.format.fill.fore_color.rgb = rgb
                marker.format.line.color.rgb      = rgb

    # ── Styling — legend ──────────────────────────────────────────────────────

    def _apply_legend(self, chart, has_series: bool, brand):
        if has_series:
            chart.has_legend              = True
            chart.legend.position         = XL_LEGEND_POSITION.BOTTOM
            chart.legend.include_in_layout = False
            self._set_font(
                chart.legend.font,
                brand.fonts.body,
                brand.fonts.caption_size,
                bold      = False,
                hex_color = brand.colors.text_dark,
                brand     = brand,
            )
        else:
            chart.has_legend = False

    # ── Styling — axes ────────────────────────────────────────────────────────

    def _apply_axes(
        self,
        chart,
        chart_type_str: str,
        x_label:        str,
        y_label:        str,
        brand,
    ):
        # Pie has no axes
        if chart_type_str == "pie":
            return

        try:
            cat_axis = chart.category_axis
            val_axis = chart.value_axis

            # Tick label fonts
            self._set_font(
                cat_axis.tick_labels.font,
                brand.fonts.body,
                brand.fonts.caption_size - 1,
                bold      = False,
                hex_color = brand.colors.text_dark,
                brand     = brand,
            )
            self._set_font(
                val_axis.tick_labels.font,
                brand.fonts.body,
                brand.fonts.caption_size - 1,
                bold      = False,
                hex_color = brand.colors.text_dark,
                brand     = brand,
            )

            # Axis titles
            if x_label:
                cat_axis.has_title = True
                cat_axis.axis_title.text_frame.text = x_label
                self._set_font(
                    cat_axis.axis_title.text_frame.paragraphs[0].add_run().font,
                    brand.fonts.body,
                    brand.fonts.caption_size,
                    bold      = False,
                    hex_color = brand.colors.text_muted,
                    brand     = brand,
                )

            if y_label:
                val_axis.has_title = True
                val_axis.axis_title.text_frame.text = y_label
                self._set_font(
                    val_axis.axis_title.text_frame.paragraphs[0].add_run().font,
                    brand.fonts.body,
                    brand.fonts.caption_size,
                    bold      = False,
                    hex_color = brand.colors.text_muted,
                    brand     = brand,
                )

        except Exception as exc:
            logger.debug("_apply_axes skipped: %s", exc)

    # ── Font helper ───────────────────────────────────────────────────────────

    @staticmethod
    def _set_font(font, name: str, size: int, bold: bool, hex_color: str, brand):
        try:
            font.name      = name
            font.size      = Pt(size)
            font.bold      = bold
            r, g, b        = brand.colors.hex_to_rgb(hex_color)
            font.color.rgb = RGBColor(r, g, b)
        except Exception as exc:
            logger.debug("_set_font skipped: %s", exc)