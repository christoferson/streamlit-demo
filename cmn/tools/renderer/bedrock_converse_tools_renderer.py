from abc import ABC, abstractmethod
from typing import Any


class AbstractToolRenderer(ABC):
    """
    Base class for all tool UI renderers.
    Keeps Streamlit code out of tool definitions.
    Each renderer knows how to display results for one tool.
    result_container is passed in — not imported — keeps renderer testable.
    """

    @property
    @abstractmethod
    def tool_name(self) -> str:
        """Must match the tool's toolSpec name exactly."""
        raise NotImplementedError

    @abstractmethod
    def render(
        self,
        tool_args:        dict,
        tool_result:      Any,
        result_container,
    ) -> None:
        raise NotImplementedError


class RendererRegistry:
    """
    Maps tool_name → renderer. O(1) lookup.

    Usage:
        registry = RendererRegistry([ChartToolRenderer(), ...])
        registry.render(tool_name, tool_args, tool_result, result_container)
    """

    def __init__(self, renderers: list[AbstractToolRenderer]):
        self._renderers = {r.tool_name: r for r in renderers}

    @property
    def renderer_names(self) -> list[str]:
        return list(self._renderers.keys())

    def render(
        self,
        tool_name:        str,
        tool_args:        dict,
        tool_result:      Any,
        result_container,
    ) -> None:
        renderer = self._renderers.get(tool_name)
        if renderer:
            renderer.render(tool_args, tool_result, result_container)
        # no match → silently skip, tool has no UI renderer