from dataclasses import dataclass, field
from typing import Optional
import json


################################################################################
# SECTION: Constants
################################################################################

STOP_REASON_MESSAGES = {
    "max_tokens":           "Insufficient Tokens. Increase MaxToken Settings.",
    "guardrail_intervened": "Response blocked by guardrail.",
    "content_filtered":     "Content was filtered.",
}

_EXCEPTION_EVENTS = [
    "internalServerException",
    "modelStreamErrorException",
    "throttlingException",
    "validationException",
    "serviceUnavailableException",
]


################################################################################
# SECTION: ToolInvocation
################################################################################

@dataclass
class ToolInvocation:
    """Captures a single tool call requested by the model."""
    tool_use_id:    str = None
    tool_name:      str = None
    tool_input_raw: str = ""        # accumulated JSON string from stream deltas
    tool_arguments: dict = None     # parsed after stream ends

    @property
    def is_pending(self) -> bool:
        return self.tool_name is not None

    def finalize(self):
        """Parse raw input string into dict. Call once after stream ends."""
        if self.tool_input_raw:
            self.tool_arguments = json.loads(self.tool_input_raw)
        else:
            self.tool_arguments = {}
        return self


################################################################################
# SECTION: StreamMetrics
################################################################################

@dataclass
class StreamMetrics:
    input_tokens:  int = 0
    output_tokens: int = 0
    total_tokens:  int = 0
    latency_ms:    int = 0


################################################################################
# SECTION: StreamResult
################################################################################

@dataclass
class StreamResult:
    """Everything extracted from one converse_stream call."""
    text:             str = ""
    stop_reason:      str = ""
    tool_invocation:  Optional[ToolInvocation] = None       # first tool (backward compat)
    tool_invocations: list = field(default_factory=list)    # all tools
    metrics:          StreamMetrics = field(default_factory=StreamMetrics)
    errors:           list = field(default_factory=list)

    @property
    def has_tool_call(self) -> bool:
        return (
            self.stop_reason == "tool_use"
            and bool(self.tool_invocations)
        )

    @property
    def stop_reason_display(self) -> Optional[str]:
        """Human-readable stop reason, or None if normal end_turn / tool_use."""
        if self.stop_reason in ("end_turn", "tool_use"):
            return None
        return STOP_REASON_MESSAGES.get(self.stop_reason, self.stop_reason)