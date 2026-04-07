"""
cmn/bedrock/converse/stream_processor.py
=========================================
Pure boto3 event stream parser for the Bedrock converse_stream API.

No boto3 client calls, no streamlit, no tool execution.
Receives the raw event stream iterator and returns a StreamResult.

Functions
---------
process_stream(stream, on_text_delta)  : main entry point — iterates events
_handle_metadata(metadata, result)     : extracts token counts and latency

Callback
--------
on_text_delta(chunk: str) -> None
    Optional callable fired for every streamed text chunk.
    Intended for live UI updates — kept as a plain callable so this
    module stays UI-agnostic.
"""

import logging
from typing import Callable, Optional

from cmn.bedrock.converse.models import (
    StreamResult,
    ToolInvocation,
    _EXCEPTION_EVENTS,
)

logger = logging.getLogger(__name__)


################################################################################
# SECTION: Internal Helpers
################################################################################

def _handle_metadata(metadata: dict, result: StreamResult) -> None:
    """
    Extract token usage and latency from the metadata event.
    Mutates result.metrics in place.
    """
    if 'usage' in metadata:
        u = metadata['usage']
        result.metrics.input_tokens  = u.get('inputTokens',  0)
        result.metrics.output_tokens = u.get('outputTokens', 0)
        result.metrics.total_tokens  = u.get('totalTokens',  0)

    if 'metrics' in metadata:
        result.metrics.latency_ms = metadata['metrics'].get('latencyMs', 0)


################################################################################
# SECTION: Main Entry Point
################################################################################

def process_stream(
    stream,
    on_text_delta: Optional[Callable[[str], None]] = None,
) -> StreamResult:
    """
    Iterate a boto3 converse_stream event stream and return a StreamResult.

    Parameters
    ----------
    stream        : the raw event stream from response['stream']
    on_text_delta : optional callback fired for each text chunk — use for
                    live UI streaming. Kept as plain callable so this
                    function stays UI-agnostic.

    Returns
    -------
    StreamResult  : fully populated after all events consumed
    """
    result           = StreamResult()
    tool_invocations = []
    current_tool     = None

    for event in stream:

        # ── Message start ─────────────────────────────────────────────────────
        if 'messageStart' in event:
            logger.debug("messageStart: role=%s",
                         event['messageStart'].get('role'))

        # ── Content block start ───────────────────────────────────────────────
        elif 'contentBlockStart' in event:
            start = event['contentBlockStart'].get('start', {})

            if 'toolUse' in start:
                tu           = start['toolUse']
                current_tool = ToolInvocation(
                    tool_use_id = tu['toolUseId'],
                    tool_name   = tu['name'],
                )
                tool_invocations.append(current_tool)
                logger.info("Tool call started: id=%s name=%s",
                            tu['toolUseId'], tu['name'])
            else:
                current_tool = None     # text block — not a tool

        # ── Content block delta ───────────────────────────────────────────────
        elif 'contentBlockDelta' in event:
            delta = event['contentBlockDelta']['delta']

            if 'text' in delta:
                chunk = delta['text']
                result.text += chunk
                if on_text_delta:
                    on_text_delta(chunk)

            elif 'toolUse' in delta:
                if current_tool is not None:
                    current_tool.tool_input_raw += delta['toolUse'].get('input', '')

            elif 'reasoningContent' in delta:
                rc = delta['reasoningContent']
                if 'text' in rc:
                    logger.debug("Reasoning delta: %s", rc['text'])

        # ── Content block stop ────────────────────────────────────────────────
        elif 'contentBlockStop' in event:
            if current_tool is not None:
                current_tool.finalize()
                logger.info("Tool call finalized: name=%s args=%s",
                            current_tool.tool_name,
                            current_tool.tool_arguments)
                current_tool = None

        # ── Message stop ──────────────────────────────────────────────────────
        elif 'messageStop' in event:
            result.stop_reason = event['messageStop'].get('stopReason', '')

            if result.stop_reason == 'tool_use' and tool_invocations:
                result.tool_invocations = tool_invocations
                result.tool_invocation  = tool_invocations[0]  # backward compat

        # ── Metadata ──────────────────────────────────────────────────────────
        elif 'metadata' in event:
            _handle_metadata(event['metadata'], result)

        # ── Stream exceptions ─────────────────────────────────────────────────
        else:
            for exc_key in _EXCEPTION_EVENTS:
                if exc_key in event:
                    msg = event[exc_key].get('message', exc_key)
                    logger.error("Stream exception [%s]: %s", exc_key, msg)
                    result.errors.append(f"[{exc_key}] {msg}")

    return result