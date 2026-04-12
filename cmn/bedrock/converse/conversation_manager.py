"""
cmn/bedrock/converse/conversation_manager.py
=============================================
Orchestrates multi-turn Bedrock conversations including tool-use loops.

All boto3 converse_stream calls are made exclusively inside _call_llm().
No streamlit imports. No rendering logic.

The UI layer supplies plain callable callbacks so this class stays
completely UI-agnostic:

    on_text_delta(chunk: str)
        Fired for every streamed text chunk.

    on_stream_result(result: StreamResult)
        Fired after each complete LLM response (may be followed by
        another LLM call if tools were invoked).

    on_tool_invoked(tool_name: str, tool_args: dict, tool_result: Any)
        Fired after each individual tool execution.

Usage
-----
    manager = ConversationManager(
        bedrock_client   = boto3.client('bedrock-runtime'),
        tool_registry    = my_registry,
        model_id         = "anthropic.claude...",
        inference_config = {"temperature": 0.1, "maxTokens": 4096},
        system_prompts   = [{"text": "You are ..."}],
    )
    result = manager.run(message_history, on_text_delta=..., ...)
"""

import logging
from typing import Any, Callable, Optional

from botocore.exceptions import ClientError

from cmn.bedrock.converse.models import StreamResult
from cmn.bedrock.converse.stream_processor import process_stream

logger = logging.getLogger(__name__)


class ConversationManager:
    """
    Orchestrates multi-turn conversations including tool-use loops.
    converse_stream is called ONLY inside _call_llm().
    """

    def __init__(
        self,
        bedrock_client,
        tool_registry,
        model_id:                str,
        inference_config:        dict,
        system_prompts:          list,
        additional_model_fields: Optional[dict] = None,
    ):
        self.client                  = bedrock_client
        self.registry                = tool_registry
        self.model_id                = model_id
        self.inference_config        = inference_config
        self.system_prompts          = system_prompts
        self.additional_model_fields = additional_model_fields

    # ── Public API ────────────────────────────────────────────────────────────

    def run(
        self,
        message_history:  list,
        on_text_delta:    Optional[Callable[[str], None]] = None,
        on_stream_result: Optional[Callable[[StreamResult], None]] = None,
        on_tool_invoked:  Optional[Callable[[str, dict, Any], None]] = None,
    ) -> StreamResult:
        """
        Run a full conversation turn, including any tool-use loops.

        Appends to a local copy of message_history — the caller's list
        is never mutated.

        Parameters
        ----------
        message_history  : existing conversation turns
        on_text_delta    : callback for each streamed text chunk
        on_stream_result : callback after each complete LLM response
        on_tool_invoked  : callback after each tool execution

        Returns
        -------
        StreamResult from the final LLM response
        """
        messages = message_history.copy()

        while True:
            result = self._call_llm(messages, on_text_delta)

            if on_stream_result:
                on_stream_result(result)

            if not result.has_tool_call:
                return result

            # ── Build assistant message for ALL tool calls ────────────────────
            assistant_content = [
                {
                    "toolUse": {
                        "toolUseId": tool_inv.tool_use_id,
                        "name":      tool_inv.tool_name,
                        "input":     tool_inv.tool_arguments,
                    }
                }
                for tool_inv in result.tool_invocations
            ]
            messages.append({"role": "assistant", "content": assistant_content})

            # ── Execute each tool + collect results ───────────────────────────
            tool_results_content = []

            for tool_inv in result.tool_invocations:
                tool_result = self.registry.invoke(
                    tool_inv.tool_name,
                    tool_inv.tool_arguments,
                )

                if on_tool_invoked:
                    on_tool_invoked(
                        tool_inv.tool_name,
                        tool_inv.tool_arguments,
                        tool_result,
                    )

                tool_results_content.append({
                    "toolResult": {
                        "toolUseId": tool_inv.tool_use_id,
                        "content":   [{"json": {"result": tool_result}}],
                    }
                })

            # ── Single user message with all tool results ─────────────────────
            messages.append({"role": "user", "content": tool_results_content})
            # loop → model sees all tool results and continues

    # ── Private ───────────────────────────────────────────────────────────────

    def _call_llm(
        self,
        messages:      list,
        on_text_delta: Optional[Callable[[str], None]],
    ) -> StreamResult:
        """
        Single converse_stream call.
        Every LLM call in this session goes through here.

        Parameters
        ----------
        messages      : full conversation history for this call
        on_text_delta : forwarded to process_stream for live streaming

        Returns
        -------
        StreamResult — errors list populated on ClientError
        """
        try:
            kwargs = dict(
                modelId         = self.model_id,
                messages        = messages,
                system          = self.system_prompts,
                toolConfig      = self.registry.tool_config,
                inferenceConfig = self.inference_config,
            )
            if self.additional_model_fields:
                kwargs['additionalModelRequestFields'] = self.additional_model_fields

            response = self.client.converse_stream(**kwargs)
            stream = response['stream']
            try:
                return process_stream(stream, on_text_delta)
            finally:
                # Ensure stream is fully consumed and closed
                if hasattr(stream, 'close'):
                    try:
                        stream.close()
                    except Exception as close_err:
                        logger.warning("Error closing stream: %s", close_err)

        except ClientError as err:
            msg = err.response["Error"]["Message"]
            logger.error("ClientError in _call_llm: %s", msg)
            r = StreamResult()
            r.errors.append(msg)
            return r