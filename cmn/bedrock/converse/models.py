"""
cmn/bedrock/converse/models.py
================================
Pure data models for the Bedrock converse streaming API.

No boto3, no streamlit, no business logic.
Only stdlib imports: dataclasses, typing, json.

Classes
-------
ToolInvocation  : captures a single tool call from the model stream
StreamMetrics   : token counts and latency from the metadata event
StreamResult    : everything extracted from one converse_stream call

Constants
---------
STOP_REASON_MESSAGES : human-readable stop reason strings
_EXCEPTION_EVENTS    : known boto3 stream exception event keys (internal)
"""

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
    """
    Captures a single tool call requested by the model.

    Lifecycle
    ---------
    1. Created on contentBlockStart when toolUse is detected
    2. tool_input_raw accumulated on each contentBlockDelta
    3. finalize() called on contentBlockStop — parses raw JSON into tool_arguments
    """
    tool_use_id:    str  = None
    tool_name:      str  = None
    tool_input_raw: str  = ""       # accumulated JSON string from stream deltas
    tool_arguments: dict = None     # parsed after stream ends

    @property
    def is_pending(self) -> bool:
        """True if a tool name has been set but not yet finalized."""
        return self.tool_name is not None

    def finalize(self):
        """
        Parse raw input string into dict.
        Call once after contentBlockStop.
        """
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
    """
    Token usage and latency extracted from the metadata event.
    Populated by stream_processor._handle_metadata().
    """
    input_tokens:  int = 0
    output_tokens: int = 0
    total_tokens:  int = 0
    latency_ms:    int = 0


################################################################################
# SECTION: StreamResult
################################################################################

@dataclass
class StreamResult:
    """
    Everything extracted from one converse_stream call.

    Populated incrementally by process_stream() as events arrive.

    Attributes
    ----------
    text             : accumulated assistant text
    stop_reason      : raw stop reason from messageStop event
    tool_invocation  : first tool call (backward compat — use tool_invocations)
    tool_invocations : all tool calls requested in this response
    metrics          : token counts + latency
    errors           : any stream exception messages collected
    """
    text:             str                      = ""
    stop_reason:      str                      = ""
    tool_invocation:  Optional[ToolInvocation] = None
    tool_invocations: list                     = field(default_factory=list)
    metrics:          StreamMetrics            = field(default_factory=StreamMetrics)
    errors:           list                     = field(default_factory=list)

    @property
    def has_tool_call(self) -> bool:
        """True when the model requested one or more tool calls."""
        return (
            self.stop_reason == "tool_use"
            and bool(self.tool_invocations)
        )

    @property
    def stop_reason_display(self) -> Optional[str]:
        """
        Human-readable stop reason for display.
        Returns None for normal termination (end_turn, tool_use).
        """
        if self.stop_reason in ("end_turn", "tool_use"):
            return None
        return STOP_REASON_MESSAGES.get(self.stop_reason, self.stop_reason)