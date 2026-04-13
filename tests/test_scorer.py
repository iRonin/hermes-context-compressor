"""Tests for ironin-compressor scorer module."""

import pytest
from hermes_context_compressor.scorer import score_turn, classify, score_all_turns, _extract_tool_and_file


class TestExtractToolAndFile:
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
        tool, path = _extract_tool_and_file(msg)
        assert tool == "write_file"
        assert path == "/src/main.py"

    def test_tool_call_without_path(self):
        msg = {
            "role": "assistant",
            "tool_calls": [{
                "function": {"name": "terminal", "arguments": '{"command": "ls"}'}
            }]
        }
        tool, path = _extract_tool_and_file(msg)
        assert tool == "terminal"
        assert path == ""

    def test_tool_result(self):
        msg = {
            "role": "tool",
            "content": "file /src/main.py written successfully"
        }
        tool, path = _extract_tool_and_file(msg)
        assert path == "/src/main.py"

    def test_empty_message(self):
        msg = {"role": "user", "content": "hello"}
        tool, path = _extract_tool_and_file(msg)
        assert tool == ""
        assert path == ""


class TestScoreTurn:
    def _msg(self, role, content, tool_calls=None, tool_call_id=None):
        m = {"role": role, "content": content}
        if tool_calls:
            m["tool_calls"] = tool_calls
        if tool_call_id:
            m["tool_call_id"] = tool_call_id
        return m

    # --- HIGH signals ---

    def test_user_directive(self):
        msg = self._msg("user", "Fix the bug in main.py and add error handling")
        assert score_turn(msg) >= 7

    def test_design_decision(self):
        msg = self._msg("assistant", "I'll use a factory pattern because it's more flexible")
        assert score_turn(msg) >= 7

    def test_error_diagnosis(self):
        msg = self._msg("assistant", "The issue was a race condition, fixed by adding a mutex")
        assert score_turn(msg) >= 7

    def test_file_success(self):
        msg = self._msg("tool", "file /src/main.py written successfully")
        assert score_turn(msg) >= 6

    def test_test_pass(self):
        msg = self._msg("tool", "12 passed, 3 skipped. All tests completed.")
        assert score_turn(msg) >= 6

    def test_specific_value(self):
        msg = self._msg("assistant", "Connect to port 8080 at 192.168.1.100")
        assert score_turn(msg) >= 6

    # --- LOW signals ---

    def test_short_text(self):
        msg = self._msg("assistant", "ok")
        assert score_turn(msg) <= 4

    def test_retry_language(self):
        msg = self._msg("assistant", "oops let me try that again")
        assert score_turn(msg) <= 4

    def test_acknowledgment(self):
        msg = self._msg("assistant", "got it, will do")
        assert score_turn(msg) <= 2

    def test_empty_content(self):
        msg = self._msg("assistant", "   ")
        assert score_turn(msg) <= 2

    # --- Tool-call bonus ---

    def test_tool_call_only_is_not_low(self):
        """Assistant message with only tool_calls (empty content) should not be LOW."""
        msg = self._msg("assistant", "", tool_calls=[{"function": {"name": "read_file", "arguments": '{"path": "/src/main.py"}'}}])
        score = score_turn(msg)
        assert score >= 4  # baseline 5 + 1 empty tc bonus + 1 tc bonus = 7? No: empty skips short text, so 5+1+1=7? Let me check...

    def test_tool_call_with_short_content(self):
        msg = self._msg("assistant", "reading", tool_calls=[{"function": {"name": "read_file", "arguments": '{"path": "/src/main.py"}'}}])
        assert score_turn(msg) >= 5  # 5 + 1 tc bonus = 6

    # --- Edge cases ---

    def test_score_bounds(self):
        msg = self._msg("assistant", "")
        assert 0 <= score_turn(msg) <= 10

    def test_high_score_cap(self):
        msg = self._msg("user", "Fix the issue because I'll use that pattern")
        assert score_turn(msg) <= 10

    def test_tool_role_neutral(self):
        msg = self._msg("tool", "directory created")
        assert 3 <= score_turn(msg) <= 7


class TestClassify:
    def test_high(self):
        assert classify(8, 7, 3) == "HIGH"

    def test_medium(self):
        assert classify(5, 7, 3) == "MEDIUM"

    def test_low(self):
        assert classify(2, 7, 3) == "LOW"

    def test_boundary_high(self):
        assert classify(7, 7, 3) == "HIGH"

    def test_boundary_medium(self):
        assert classify(3, 7, 3) == "MEDIUM"

    def test_boundary_low(self):
        assert classify(2, 7, 3) == "LOW"


class TestScoreAllTurns:
    def _msg(self, role, content):
        return {"role": role, "content": content}

    def test_basic_scoring(self):
        messages = [
            self._msg("system", "You are helpful"),
            self._msg("user", "Fix the bug in main.py"),
            self._msg("assistant", "I'll use a decorator pattern"),
            self._msg("user", "ok"),
        ]
        results = score_all_turns(messages, head_end=1, tail_start=3, keep_thresh=7, drop_thresh=3)
        assert len(results) == 2

    def test_empty_region(self):
        messages = [self._msg("user", "hello")]
        results = score_all_turns(messages, head_end=0, tail_start=0)
        assert results == []
