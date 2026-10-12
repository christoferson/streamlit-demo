"""
Tests for the UI-agnostic converse helpers in cmn/bedrock/converse.

Run from the project root:  python -m unittest discover -s tests -v
"""
import unittest

from cmn.bedrock_models import FOUNDATION_MODELS
from cmn.bedrock.converse import (
    InferenceSettings, build_converse_request, process_stream,
    strip_binary_content, trim_conversation_history, validate_conversation_structure,
)

ALL_SETTINGS = InferenceSettings(max_tokens=1000, temperature=0.3, top_p=0.9, top_k=40)
MESSAGES = [{"role": "user", "content": [{"text": "hi"}]}]


def request_for(model_id, settings=ALL_SETTINGS):
    return build_converse_request(FOUNDATION_MODELS.find(model_id), MESSAGES, "Be brief.", settings)


class BuildConverseRequestTest(unittest.TestCase):

    def test_common_fields(self):
        request = request_for("global.anthropic.claude-opus-5-5")
        self.assertEqual(request["modelId"], "global.anthropic.claude-opus-5-5")
        self.assertIs(request["messages"], MESSAGES)
        self.assertEqual(request["system"], [{"text": "Be brief."}])

    def test_claude_5_sends_only_max_tokens(self):
        request = request_for("global.anthropic.claude-sonnet-5-5")
        self.assertEqual(request["inferenceConfig"], {"maxTokens": 1000})
        self.assertNotIn("additionalModelRequestFields", request)

    def test_sonnet_4_6_sends_temperature_and_top_k_but_not_top_p(self):
        request = request_for("global.anthropic.claude-sonnet-4-6")
        self.assertEqual(request["inferenceConfig"], {"maxTokens": 1000, "temperature": 0.3})
        self.assertEqual(request["additionalModelRequestFields"], {"top_k": 40})

    def test_top_k_uses_the_model_native_field_name(self):
        self.assertEqual(request_for("cohere.command-r-v1:0")["additionalModelRequestFields"], {"k": 40})
        self.assertEqual(request_for("mistral.mistral-large-3-675b-instruct")["additionalModelRequestFields"], {"TopK": 40})

    def test_all_sampling_parameters(self):
        request = request_for("us.writer.palmyra-x5-v1:0")
        self.assertEqual(request["inferenceConfig"], {"maxTokens": 1000, "temperature": 0.3, "topP": 0.9})
        self.assertEqual(request["additionalModelRequestFields"], {"top_k": 40})

    def test_unset_values_are_not_sent(self):
        request = request_for("us.writer.palmyra-x5-v1:0", InferenceSettings(max_tokens=500))
        self.assertEqual(request["inferenceConfig"], {"maxTokens": 500})
        self.assertNotIn("additionalModelRequestFields", request)


class ProcessStreamTest(unittest.TestCase):

    def test_text_and_live_callback(self):
        chunks = []
        result = process_stream([
            {"messageStart": {"role": "assistant"}},
            {"contentBlockDelta": {"delta": {"text": "Hel"}}},
            {"contentBlockDelta": {"delta": {"text": "lo"}}},
            {"messageStop": {"stopReason": "end_turn"}},
            {"metadata": {"usage": {"inputTokens": 5, "outputTokens": 2, "totalTokens": 7}, "metrics": {"latencyMs": 120}}},
        ], on_text_delta=chunks.append)
        self.assertEqual(result.text, "Hello")
        self.assertEqual(chunks, ["Hel", "lo"])
        self.assertEqual((result.metrics.input_tokens, result.metrics.output_tokens, result.metrics.total_tokens), (5, 2, 7))
        self.assertEqual(result.metrics.latency_ms, 120)
        self.assertEqual(result.errors, [])
        self.assertIsNone(result.stop_reason_display)

    def test_metadata_without_usage_or_metrics(self):
        # Review bug B1: the page used to crash with NameError here
        result = process_stream([{"contentBlockDelta": {"delta": {"text": "x"}}}, {"metadata": {}}])
        self.assertEqual(result.metrics.total_tokens, 0)
        self.assertEqual(result.metrics.latency_ms, 0)

    def test_stream_error_is_kept_out_of_the_text(self):
        # Review bug B2: errors used to be appended to the saved reply
        result = process_stream([
            {"contentBlockDelta": {"delta": {"text": "partial"}}},
            {"throttlingException": {"message": "Too many requests"}},
        ])
        self.assertEqual(result.text, "partial")
        self.assertEqual(result.errors, ["[throttlingException] Too many requests"])

    def test_reasoning_deltas_are_not_text(self):
        result = process_stream([
            {"contentBlockDelta": {"delta": {"reasoningContent": {"text": "thinking..."}}}},
            {"contentBlockDelta": {"delta": {"text": "answer"}}},
        ])
        self.assertEqual(result.text, "answer")

    def test_max_tokens_stop_reason_is_explained(self):
        result = process_stream([{"messageStop": {"stopReason": "max_tokens"}}])
        self.assertIn("Insufficient Tokens", result.stop_reason_display)


def history(length):
    return [{"role": "user" if i % 2 == 0 else "assistant", "content": [{"text": str(i)}]} for i in range(length)]


class TrimConversationHistoryTest(unittest.TestCase):

    def test_under_the_limit_is_unchanged(self):
        messages = history(158)
        self.assertIs(trim_conversation_history(messages, 160), messages)

    def test_trims_to_the_limit_keeping_the_newest(self):
        for length in (161, 162, 170, 320):
            trimmed = trim_conversation_history(history(length), 160)
            with self.subTest(length=length):
                self.assertLessEqual(len(trimmed), 160)
                self.assertGreaterEqual(len(trimmed), 159)
                self.assertEqual(trimmed[0]["role"], "user")
                self.assertEqual(trimmed[-1]["content"][0]["text"], str(length - 1))


class ValidateConversationStructureTest(unittest.TestCase):

    def test_valid_conversation(self):
        validate_conversation_structure(history(4), 160)

    def test_invalid_conversations(self):
        cases = {
            "not a list": {"role": "user"},
            "too long": history(162),
            "bad role": [{"role": "robot", "content": [{"text": "x"}]}],
            "missing content": [{"role": "user"}],
            "content not a list": [{"role": "user", "content": "x"}],
            "item without text": [{"role": "user", "content": [{"image": {}}]}],
        }
        for name, conversation in cases.items():
            with self.subTest(name), self.assertRaises(ValueError):
                validate_conversation_structure(conversation, 160)


class StripBinaryContentTest(unittest.TestCase):

    def test_attachments_become_placeholders(self):
        messages = [{"role": "user", "content": [
            {"text": "look"},
            {"image": {"format": "png", "source": {"bytes": b"\x89PNG"}}},
            {"document": {"format": "pdf", "name": "report_pdf", "source": {"bytes": b"%PDF"}}},
        ]}]
        self.assertEqual(strip_binary_content(messages), [{"role": "user", "content": [
            {"text": "look"},
            {"text": "[image attachment omitted]"},
            {"text": "[document attachment omitted: report_pdf]"},
        ]}])

    def test_stripped_history_passes_validation(self):
        messages = [{"role": "user", "content": [{"text": "x"}, {"image": {"source": {"bytes": b""}}}]}]
        validate_conversation_structure(strip_binary_content(messages), 160)


if __name__ == "__main__":
    unittest.main()
