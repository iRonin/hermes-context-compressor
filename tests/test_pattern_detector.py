"""Tests for ironin-compressor pattern detector."""

import pytest
from hermes_context_compressor.pattern_detector import detect_retry_loops, collapse_retry_loops, _extract_tool_file_pair


class TestExtractToolFilePair:
    def test_tool_call_with_path(self):
        msg = {
            "role": "assistant",
            "tool_calls": [{
                "function": {
                    "name": "write_file",
                    "arguments": '{"path": "/src/main.py", "content": "hello"}'
                }
            }]
        }
        tool, path = _extract_tool_file_pair(msg)
        assert tool == "write_file"
        assert path == "/src/main.py"

    def test_no_tool_calls(self):
        msg = {"role": "user", "content": "hello"}
        tool, path = _extract_tool_file_pair(msg)
        assert tool == ""
        assert path == ""


class TestDetectRetryLoops:
    def _msg(self, role, content="", tool_calls=None):
        m = {"role": role, "content": content}
        if tool_calls:
            m["tool_calls"] = tool_calls
        return m

    def _tool_call(self, name, path):
        return [{
            "function": {
                "name": name,
                "arguments": f'{{"path": "{path}", "content": "..."}}'
            }
        }]

    def test_no_loops_single_write(self):
        messages = [
            self._msg("assistant", "", self._tool_call("write_file", "/src/main.py")),
            self._msg("tool", "file written successfully"),
        ]
        loops = detect_retry_loops(messages, window=3)
        assert loops == []

    def test_detect_write_error_write_loop(self):
        messages = [
            self._msg("assistant", "", self._tool_call("write_file", "/src/main.py")),
            self._msg("tool", "error: permission denied"),
            self._msg("assistant", "let me try with sudo", self._tool_call("write_file", "/src/main.py")),
            self._msg("tool", "file written successfully"),
        ]
        loops = detect_retry_loops(messages, window=3)
        assert len(loops) == 1
        start, end, desc = loops[0]
        assert start == 0
        assert end >= 3
        assert "write_file" in desc

    def test_different_files_no_loop(self):
        messages = [
            self._msg("assistant", "", self._tool_call("write_file", "/src/a.py")),
            self._msg("tool", "ok"),
            self._msg("assistant", "", self._tool_call("write_file", "/src/b.py")),
            self._msg("tool", "ok"),
        ]
        loops = detect_retry_loops(messages, window=3)
        assert loops == []

    def test_different_tools_no_loop(self):
        messages = [
            self._msg("assistant", "", self._tool_call("write_file", "/src/main.py")),
            self._msg("tool", "ok"),
            self._msg("assistant", "", self._tool_call("read_file", "/src/main.py")),
        ]
        loops = detect_retry_loops(messages, window=3)
        assert loops == []

    def test_empty_messages(self):
        assert detect_retry_loops([]) == []


class TestCollapseRetryLoops:
    def _msg(self, role, content="", tool_calls=None):
        m = {"role": role, "content": content}
        if tool_calls:
            m["tool_calls"] = tool_calls
        return m

    def test_collapse_with_loops(self):
        messages = [
            self._msg("assistant", "trying write", [{"function": {"name": "write_file", "arguments": '{"path": "/src/main.py"}'}}]),
            self._msg("tool", "error: permission denied"),
            self._msg("assistant", "retrying", [{"function": {"name": "write_file", "arguments": '{"path": "/src/main.py"}'}}]),
            self._msg("tool", "file written"),
        ]
        loops = [(0, 4, "write_file loop")]
        result = collapse_retry_loops(messages, loops)
        assert len(result) <= len(messages)
        assert "Retry loop detected" in result[0]["content"]

    def test_no_loops_passthrough(self):
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        result = collapse_retry_loops(messages, [])
        assert result == messages

    def test_empty_messages(self):
        assert collapse_retry_loops([], []) == []
