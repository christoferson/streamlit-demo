"""
Tests for pages/3_7_4_converse_tool_demo.py: the tool loop, history, limits, tool checkboxes and
error handling. The page runs in streamlit.testing.v1.AppTest with boto3.client patched to a fake
bedrock-runtime client, so there are no AWS calls.

Run from the project root:  python -m unittest discover -s tests -v
(Skipped on the older Python 3.8 / Streamlit 1.40 install, which can't run this page.)
"""
import inspect
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import streamlit as st
from botocore.exceptions import ClientError, EventStreamError, ReadTimeoutError

PAGE = str(Path(__file__).resolve().parent.parent / "pages" / "3_7_4_converse_tool_demo.py")
TOOLS = ["expr_evaluator", "acronym_evaluator", "url_content_loader", "wikipedia_loader", "datetime_tool"]

try:
    from streamlit.testing.v1 import AppTest
    import cmn.tools.tool  # noqa: F401  (the page imports it)
    # The page uses horizontal containers (newer Streamlit)
    PAGE_RUNNABLE = "horizontal" in inspect.signature(st.container).parameters
    SKIP_REASON = "Streamlit too old for this page"
except Exception as err:  # e.g. Python 3.8: cmn/tools/tool imports modules with 3.10+ syntax
    PAGE_RUNNABLE, SKIP_REASON = False, f"page dependencies not importable: {err}"


# ── Fake converse_stream events ──────────────────────────────────────────────

def text(index, value):
    return [{"contentBlockDelta": {"contentBlockIndex": index, "delta": {"text": value}}},
            {"contentBlockStop": {"contentBlockIndex": index}}]


def tool(index, tool_use_id, name, tool_input):
    return [{"contentBlockStart": {"contentBlockIndex": index, "start": {"toolUse": {"toolUseId": tool_use_id, "name": name}}}},
            {"contentBlockDelta": {"contentBlockIndex": index, "delta": {"toolUse": {"input": tool_input}}}},
            {"contentBlockStop": {"contentBlockIndex": index}}]


def reasoning(index, signature):
    return [{"contentBlockDelta": {"contentBlockIndex": index, "delta": {"reasoningContent": {"text": ""}}}},
            {"contentBlockDelta": {"contentBlockIndex": index, "delta": {"reasoningContent": {"signature": signature}}}},
            {"contentBlockStop": {"contentBlockIndex": index}}]


def response(*blocks, stop_reason="end_turn"):
    """A complete stream: the given content blocks, then messageStop and metadata."""
    events = [{"messageStart": {"role": "assistant"}}] + [event for block in blocks for event in block]
    return events + [{"messageStop": {"stopReason": stop_reason}},
                     {"metadata": {"usage": {"inputTokens": 10, "outputTokens": 5, "totalTokens": 15}, "metrics": {"latencyMs": 100}}}]


def broken(events, error):
    """A stream that yields some events, then raises like botocore does mid-stream."""
    def generate():
        yield from events
        raise error
    return generate


THROTTLE = EventStreamError({"Error": {"Code": "throttlingException", "Message": "Too many requests"}}, "ConverseStream")
TIMEOUT = ReadTimeoutError(endpoint_url="https://bedrock-runtime.us-east-1.amazonaws.com")


@unittest.skipUnless(PAGE_RUNNABLE, SKIP_REASON)
class ConverseToolDemoTest(unittest.TestCase):

    def setUp(self):
        self.responses = []   # what each converse_stream call returns: an event list, a generator function, or an exception
        self.requests = []    # kwargs of every converse_stream call
        client = MagicMock()
        client.converse_stream.side_effect = self._converse_stream
        st.cache_resource.clear()  # get_bedrock_runtime() is cached across runs
        patcher = patch("boto3.client", return_value=client)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.app = AppTest.from_file(PAGE, default_timeout=60)
        self.rendered = False

    def _converse_stream(self, **kwargs):
        self.requests.append(kwargs)
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return {"stream": item() if callable(item) else iter(item)}

    def render(self):
        self.app.run()
        self.rendered = True

    def send(self, prompt, *responses):
        self.responses.extend(responses)
        if not self.rendered:  # the chat input exists only after the first run
            self.render()
        self.app.chat_input[0].set_value(prompt).run()
        self.assertFalse(self.app.exception, [e.value for e in self.app.exception])

    @property
    def history(self):
        return self.app.session_state["menu_converse_tool_messages"]

    @property
    def tool_log(self):
        return self.app.session_state["menu_converse_tool_log"]

    def bubbles(self):
        return [(m.name, " ".join(md.value for md in m.markdown)) for m in self.app.chat_message]

    def counter(self):
        return next(m.value for m in self.app.markdown if "/200" in m.value)

    def block_types(self, message):
        return [next(iter(block)) for block in message["content"]]

    # ── Tool loop ─────────────────────────────────────────────────────────────

    def test_plain_answer(self):
        self.send("Hi", response(text(0, "Hello!")))
        self.assertEqual([(m["role"], m["content"][0]["text"]) for m in self.history], [("user", "Hi"), ("assistant", "Hello!")])
        self.assertIn("2/200", self.counter())

    def test_parallel_tools_then_follow_up_with_thinking(self):
        self.send(
            "6*7 and the date?",
            response(tool(0, "t1", "expr_evaluator", '{"expression": "6 * 7"}'),
                     tool(1, "t2", "datetime_tool", '{"expression": "current_date"}'), stop_reason="tool_use"),
            response(reasoning(0, "SIG"), text(1, "Checking once more."),
                     tool(2, "t3", "expr_evaluator", '{"expression": "2 + 2"}'), stop_reason="tool_use"),
            response(text(0, "42, and 4.")),
        )
        self.assertEqual(len(self.requests), 3)
        # Every toolUse of a response gets a toolResult in the next request
        second_request = self.requests[1]["messages"]
        self.assertEqual([c["toolResult"]["toolUseId"] for c in second_request[-1]["content"]], ["t1", "t2"])
        self.assertEqual(second_request[-1]["content"][0]["toolResult"]["content"][0]["json"], {"expr_result": 42})
        # Thinking goes back with its signature while the tool loop runs
        third_request = self.requests[2]["messages"]
        self.assertEqual(self.block_types(third_request[-2]), ["reasoningContent", "text", "toolUse"])
        self.assertEqual(third_request[-2]["content"][0]["reasoningContent"]["reasoningText"]["signature"], "SIG")
        # History keeps the tool exchange (not the thinking); the chat shows one bubble per turn
        self.assertEqual([self.block_types(m) for m in self.history],
                         [["text"], ["toolUse", "toolUse"], ["toolResult", "toolResult"], ["text", "toolUse"], ["toolResult"], ["text"]])
        self.assertEqual([name for name, _ in self.bubbles()], ["user", "assistant"])
        self.assertEqual([entry["tool_name"] for entry in self.tool_log], ["expr_evaluator", "datetime_tool", "expr_evaluator"])
        self.assertIn("2/200", self.counter())  # visible messages, not the 6 in history

    def test_tool_rounds_are_capped(self):
        endless = [response(tool(0, f"t{n}", "expr_evaluator", '{"expression": "1 + 1"}'), stop_reason="tool_use") for n in range(10)]
        self.send("Loop", *endless)
        self.assertEqual(len(self.requests), 6)  # 5 tool rounds, then the 6th request is not run
        self.assertIn("Not run", str(self.tool_log[-1]["result"]))
        self.assertEqual(self.block_types(self.history[-1]), ["text"])  # no toolUse without a result

    def test_invalid_tool_input_still_gets_a_result(self):
        self.send("Bad input", response(tool(0, "t1", "expr_evaluator", '{"expression": '), stop_reason="tool_use"),
                  response(text(0, "Sorry.")))
        self.assertEqual(self.requests[1]["messages"][-2]["content"][0]["toolUse"]["input"], {"invalid_json": '{"expression": '})

    # ── History limit and display ─────────────────────────────────────────────

    def test_history_is_trimmed_to_the_visible_limit(self):
        def tool_turn(n):
            return [{"role": "user", "content": [{"text": f"question {n}"}]},
                    {"role": "assistant", "content": [{"toolUse": {"toolUseId": f"t{n}", "name": "expr_evaluator", "input": {}}}]},
                    {"role": "user", "content": [{"toolResult": {"toolUseId": f"t{n}", "content": [{"json": {"expr_result": 2}}]}}]},
                    {"role": "assistant", "content": [{"text": f"answer {n}"}]}]
        self.app.session_state["menu_converse_tool_messages"] = [m for n in range(100) for m in tool_turn(n)]
        self.render()
        self.assertIn("200/200", self.counter())
        self.send("new question", response(text(0, "new answer")))
        self.assertIn("200/200", self.counter())
        self.assertEqual(self.history[0], {"role": "user", "content": [{"text": "question 1"}]})  # oldest whole turn dropped
        self.assertEqual(self.history[-1]["content"][0]["text"], "new answer")

    def test_clear_conversation(self):
        self.send("6*7?", response(tool(0, "t1", "expr_evaluator", '{"expression": "6 * 7"}'), stop_reason="tool_use"),
                  response(text(0, "42")))
        self.app.button(key="clear_conversation").click().run()
        self.assertEqual(self.history, [])
        self.assertEqual(self.tool_log, [])
        self.assertNotIn("menu_converse_tool_last_metrics", self.app.session_state)
        self.assertIn("0/200", self.counter())

    # ── Tool checkboxes ───────────────────────────────────────────────────────

    def offered_tools(self, request):
        return [t["toolSpec"]["name"] for t in request.get("toolConfig", {}).get("tools", [])]

    def test_unticked_tool_is_not_offered_or_run(self):
        self.render()
        self.app.sidebar.checkbox(key="tool_enabled_expr_evaluator").uncheck().run()
        self.send("6*7?", response(tool(0, "t1", "expr_evaluator", '{"expression": "6 * 7"}'), stop_reason="tool_use"),
                  response(text(0, "Worked it out by hand: 42.")))
        self.assertNotIn("expr_evaluator", self.offered_tools(self.requests[0]))
        self.assertEqual(self.tool_log[0]["result"], {"error": "Tool expr_evaluator is disabled"})
        # Availability is recorded on the user message (hidden in the chat) and explained in the system prompt
        self.assertTrue(any("Available now" in block["text"] for block in self.requests[0]["system"]))
        self.assertTrue(self.history[0]["content"][-1]["text"].startswith("[Tools available for this message:"))
        self.assertEqual(self.bubbles()[0], ("user", "6*7?"))

    def test_all_tools_unticked_sends_history_without_tool_blocks(self):
        self.send("6*7?", response(tool(0, "t1", "expr_evaluator", '{"expression": "6 * 7"}'), stop_reason="tool_use"),
                  response(text(0, "42")))
        for name in TOOLS:
            self.app.sidebar.checkbox(key=f"tool_enabled_{name}").uncheck()
        self.app.run()
        self.send("And 7*8?", response(text(0, "56")))
        request = self.requests[-1]
        self.assertNotIn("toolConfig", request)
        blocks = [block for message in request["messages"] for block in message["content"]]
        self.assertFalse(any("toolUse" in block or "toolResult" in block for block in blocks))
        self.assertTrue(any(block.get("text", "").startswith("[Called tool expr_evaluator") for block in blocks))

    # ── Errors ────────────────────────────────────────────────────────────────

    def assert_failed_turn_not_saved(self):
        errors = [text for name, text in self.bubbles() if name == "system"]
        self.assertTrue(errors and errors[0].startswith(":red["), self.bubbles())
        self.assertEqual(self.history, [])
        # The next message works, so the history stayed valid
        self.send("Try again", response(text(0, "Recovered")))
        self.assertEqual([m["content"][0]["text"] for m in self.history], ["Try again", "Recovered"])

    def test_client_error_on_call(self):
        self.send("Hi", ClientError({"Error": {"Code": "ValidationException", "Message": "Bad request"}}, "ConverseStream"))
        self.assert_failed_turn_not_saved()

    def test_read_timeout_on_call(self):
        self.send("Hi", TIMEOUT)
        self.assert_failed_turn_not_saved()

    def test_throttle_mid_stream(self):
        self.send("Hi", broken(text(0, "Partial ans")[:1], THROTTLE))
        self.assert_failed_turn_not_saved()

    def test_stream_breaks_during_tool_request(self):
        half_tool = tool(0, "t1", "expr_evaluator", '{"expre')[:2]
        self.send("6*7?", broken(half_tool, THROTTLE))
        self.assertEqual(self.tool_log, [])  # the half-received tool request is not run
        self.assertEqual(len(self.requests), 1)
        self.assert_failed_turn_not_saved()

    def test_timeout_in_follow_up_call(self):
        self.send("6*7?", response(tool(0, "t1", "expr_evaluator", '{"expression": "6 * 7"}'), stop_reason="tool_use"), TIMEOUT)
        self.assertEqual(len(self.tool_log), 1)  # the tool did run before the follow-up failed
        self.assert_failed_turn_not_saved()


if __name__ == "__main__":
    unittest.main()
