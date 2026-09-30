"""A tool call written inside the thinking (no </think>) must come out as a tool call, not as reasoning text.

    python -m unittest serve.test_reasoning_call -v
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.frontend import OutputParser  # noqa: E402

TOOLS = [
    {"name": "grep_search", "parameters": {"type": "object", "properties": {
        "query": {"type": "string"}, "isRegexp": {"type": "boolean"}, "maxResults": {"type": "number"}}}},
    {"name": "read_file", "parameters": {"type": "object", "properties": {
        "filePath": {"type": "string"}, "startLine": {"type": "number"}, "endLine": {"type": "number"}}}},
]

# The shape of the 2026-09-30 Copilot failure: reasoning, then two calls, then the end of the turn - no </think>.
SKIPPED_THINK_END = (
    "I need to check how GenericFlash allocates sectors.\n\n"
    "<tool_call>\n<function=grep_search>\n<parameter=query>\nsectorSize|numberOfSectors\n</parameter>\n"
    "<parameter=isRegexp>\ntrue\n</parameter>\n<parameter=maxResults>\n30\n</parameter>\n</function>\n</tool_call>\n"
    "<tool_call>\n<function=read_file>\n<parameter=filePath>\nd:\\fdna\\CardCfgData.c\n</parameter>\n"
    "<parameter=startLine>\n500\n</parameter>\n<parameter=endLine>\n560\n</parameter>\n</function>\n</tool_call>"
)


def run(text, tools=TOOLS, stream_tools=False, step=None):
    p = OutputParser(thinking=True, tools=tools, stream_tools=stream_tools)
    events = []
    step = step or len(text)
    for i in range(0, len(text), step):
        events += p.feed(text[i:i + step])
    events += p.finish()
    reasoning = "".join(e.text for e in events if e.kind == "reasoning")
    content = "".join(e.text for e in events if e.kind == "content")
    calls = [(e.call.name, e.call.arguments) for e in events if e.kind == "tool_call"]
    return reasoning, content, calls


class ReasoningToolCall(unittest.TestCase):
    def test_calls_inside_thinking_are_calls(self):
        for stream_tools in (False, True):
            for step in (None, 1, 3, 7):
                with self.subTest(stream_tools=stream_tools, step=step):
                    reasoning, content, calls = run(SKIPPED_THINK_END, stream_tools=stream_tools, step=step)
                    self.assertEqual(reasoning.strip(), "I need to check how GenericFlash allocates sectors.")
                    self.assertEqual(content, "")
                    self.assertEqual(calls, [
                        ("grep_search", {"query": "sectorSize|numberOfSectors", "isRegexp": True, "maxResults": 30}),
                        ("read_file", {"filePath": "d:\\fdna\\CardCfgData.c", "startLine": 500, "endLine": 560}),
                    ])

    def test_prose_mention_stays_reasoning(self):
        text = "The template wraps calls in <tool_call> tags, so </think>Done."
        for step in (None, 1):
            reasoning, content, calls = run(text, step=step)
            self.assertEqual(reasoning, "The template wraps calls in <tool_call> tags, so ")
            self.assertEqual((content, calls), ("Done.", []))

    def test_without_tools_nothing_changes(self):
        reasoning, content, calls = run(SKIPPED_THINK_END, tools=None, step=5)
        self.assertEqual((reasoning, content, calls), (SKIPPED_THINK_END, "", []))

    def test_normal_think_end_unchanged(self):
        text = "plan\n</think>\n\n" + SKIPPED_THINK_END.split("\n\n", 1)[1]
        for step in (None, 1):
            reasoning, content, calls = run(text, step=step)
            self.assertEqual((reasoning, content), ("plan\n", ""))
            self.assertEqual([c[0] for c in calls], ["grep_search", "read_file"])

    def test_ends_on_a_partial_tag(self):
        reasoning, content, calls = run("thinking <tool_call>\n<func", step=2)
        self.assertEqual(reasoning, "thinking <tool_call>\n<func")
        self.assertEqual((content, calls), ("", []))


if __name__ == "__main__":
    unittest.main()
