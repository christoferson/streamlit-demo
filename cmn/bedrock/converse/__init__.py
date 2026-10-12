"""
cmn/bedrock/converse
=====================
Public API for the Bedrock converse streaming package.

Typical import
--------------
    from cmn.bedrock.converse import ConversationManager, StreamResult
"""

from cmn.bedrock.converse.models import (
    ToolInvocation,
    StreamMetrics,
    StreamResult,
    STOP_REASON_MESSAGES,
    _EXCEPTION_EVENTS,
)
from cmn.bedrock.converse.stream_processor import process_stream
from cmn.bedrock.converse.conversation_manager import ConversationManager
from cmn.bedrock.converse.request_builder import (
    InferenceSettings,
    build_converse_request,
)
from cmn.bedrock.converse.conversation_history import (
    strip_binary_content,
    trim_conversation_history,
    validate_conversation_structure,
)