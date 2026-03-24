from abc import ABC, abstractmethod
from typing import Any
import json
import logging

logger = logging.getLogger(__name__)


class AbstractBedrockConverseTool(ABC):
    """
    Base class for all Bedrock converse tools.
    """
    definition: object

    def __init__(self, name, definition):
        self.name       = name
        self.definition = definition

    def matches(self, name):
        return self.name == name

    def summary(self) -> str | None:
        """
        Override to provide a one-line description for the system prompt.
        Returns None by default — excluded from system prompt if not overridden.
        """
        return None

    @abstractmethod
    def invoke(self, params, tool_args=None):
        raise NotImplementedError


class ToolRegistry:
    """
    Registers tools and provides O(1) lookup + dispatch.

    Usage:
        registry = ToolRegistry([calc_tool, wiki_tool, ...])
        tool_cfg = registry.tool_config       # pass to converse_stream
        result   = registry.invoke(name, args)
    """

    def __init__(self, tools: list[AbstractBedrockConverseTool]):
        self._tools = {
            tool.definition['toolSpec']['name']: tool
            for tool in tools
        }

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def tool_config(self) -> dict:
        """Returns toolConfig dict ready for converse_stream."""
        return {"tools": [t.definition for t in self._tools.values()]}

    @property
    def tool_names(self) -> list[str]:
        return list(self._tools.keys())

    def build_tool_summary(self) -> str | None:
        """
        Loop through all tools, collect non-null summaries,
        return formatted string or None if no summaries exist.
        """
        lines = [
            tool.summary()
            for tool in self._tools.values()
            if tool.summary() is not None
        ]

        if not lines:
            return None

        return (
            "AVAILABLE TOOLS:\n"
            + "\n".join(f"  - {line}" for line in lines)
        )

    # ── Dispatch ──────────────────────────────────────────────────────────────

    def invoke(self, tool_name: str, tool_args: dict) -> Any:
        """
        Dispatch to the matching tool.
        Raises KeyError if tool_name is not registered.
        """
        if tool_name not in self._tools:
            raise KeyError(
                f"Unknown tool: '{tool_name}'. Available: {self.tool_names}"
            )
        tool = self._tools[tool_name]
        logger.info("Invoking tool '%s' with args: %s", tool_name, tool_args)
        return tool.invoke(params=None, tool_args=tool_args)