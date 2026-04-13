"""Heuristic turn scorer for ironin-compressor.

Scores each message turn on a 0-10 scale based on content signals.
HIGH (>=7): keep verbatim.  MEDIUM (4-6): summarize.  LOW (<4): drop.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Signal definitions
# ---------------------------------------------------------------------------

# HIGH signals — things that indicate substantive content
_HIGH_SIGNALS = [
    # (name, regex, weight, roles_to_check)
    ("user_directive", r"\b(do|change|update|fix|create|add|remove|delete|implement|rewrite|refactor|configure)\b", 3, ("user",)),
    ("design_decision", r"\b(I['']ll use|using|I chose|decided to|pattern|approach|strategy|because|since|therefore)\b", 3, ("assistant",)),
    ("error_diagnosis", r"\b(issue was|problem was|root cause|error was|the bug|fixed by|the fix|resolved by|caused by)\b", 3, ("assistant",)),
    ("file_success", r"\b(file written|created|updated|modified|saved successfully)\b", 2, ("tool",)),
    ("test_pass", r"\b(\d+ passed|all tests passed|test.*pass|PASSED)\b", 2, ("tool",)),
    ("specific_value", r"([/\\][\w.-]+|line \d+|error \d+|\b0x[0-9a-f]+\b|\b\d+\.\d+\.\d+\b)", 2, ("user", "assistant", "tool")),
]

# LOW signals — things that indicate low-value content
_LOW_SIGNALS = [
    # (name, regex, weight)
    ("short_text", None, -1),  # handled specially: len < 50
    ("retry_language", r"\b(let me try|oops|actually|wait|let['']s try|retrying|trying again|give me another|one more)\b", -2),
    ("tool_error_retry", r"\b(error|failed|crashed|timeout|exception|traceback)(:| occurred| detected| found| thrown| caught| message)\b", -2),
    ("acknowledgment", r"\b(got it|understood|sure|okay|ok|will do|on it|let me)\b", -2),
    ("empty_content", None, -3),  # handled specially: empty/whitespace
]

# Retry pattern: same tool + same file within a window
_TOOL_CALL_RE = re.compile(r'["\']?name["\']?\s*[:=]\s*["\'](\w+)["\']')
_FILE_PATH_RE = re.compile(r'["\']?path["\']?\s*[:=]\s*["\']([/\w.\-]+)["\']')


def _extract_tool_and_file(msg: Dict[str, Any]) -> tuple[str, str]:
    """Extract tool name and file path from a message for retry detection."""
    tool_name = ""
    file_path = ""

    # Check tool_calls
    for tc in msg.get("tool_calls") or []:
        if isinstance(tc, dict):
            fn = tc.get("function", {})
            tool_name = fn.get("name", "")
            args = fn.get("arguments", "")
            m = _FILE_PATH_RE.search(args)
            if m:
                file_path = m.group(1)

    # Check tool result content for file path
    if msg.get("role") == "tool":
        content = msg.get("content", "")
        m = re.search(r'file\s+(\S+)', content)
        if m:
            file_path = m.group(1)

    return tool_name, file_path


def score_turn(
    msg: Dict[str, Any],
    prev_msgs: Optional[List[Dict[str, Any]]] = None,
    turn_index: int = 0,
    window: int = 3,
) -> int:
    """Score a single message turn from 0 to 10.

    Args:
        msg: The message to score.
        prev_msgs: Recent messages before this one (for context-aware scoring).
        turn_index: Position in the conversation (0 = first after head).
        window: Look-back window for retry detection.
    """
    score = 5  # neutral baseline
    role = msg.get("role", "unknown")
    content = msg.get("content") or ""

    # Empty content → penalty, but NOT for tool-calling messages
    # (tool-call-only messages are common and substantive)
    if not content.strip():
        if msg.get("tool_calls"):
            score += 1  # bonus for active tool work
        else:
            return max(0, score - 3)

    # Short text penalty — but NOT for assistant messages with tool_calls
    # (tool-invocation messages are inherently short but substantive)
    if role in ("user", "assistant") and len(content.strip()) < 50:
        if not (role == "assistant" and msg.get("tool_calls")):
            score -= 1

    # Tool-calling bonus — differentiate write (substantive) vs read (expected)
    if role == "assistant" and msg.get("tool_calls"):
        for tc in msg.get("tool_calls") or []:
            if isinstance(tc, dict):
                tname = tc.get("function", {}).get("name", "")
                if tname in ("write_file", "patch"):
                    score += 2  # writing/modifying files is substantive work
                elif tname in ("read_file", "search_files"):
                    pass  # reading is expected, no bonus

    # HIGH signals
    for name, regex, weight, roles in _HIGH_SIGNALS:
        if role not in roles:
            continue
        if regex is None:
            continue
        if re.search(regex, content, re.IGNORECASE):
            score += weight

    # LOW signals
    for name, regex, weight in _LOW_SIGNALS:
        if regex is None:
            if name == "short_text":
                continue  # already counted above
            if name == "empty_content" and not content.strip():
                # Only penalize empty content if there are no tool_calls
                # (tool-call-only messages are substantive work)
                if not msg.get("tool_calls"):
                    score += weight
            continue
        if re.search(regex, content, re.IGNORECASE):
            score += weight

    # Context-aware: tool error retry detection
    if role == "assistant" and prev_msgs:
        for prev in prev_msgs[-window:]:
            if prev.get("role") == "tool":
                tool_content = prev.get("content", "")
                if re.search(r'\b(error|failed|crashed|exception|traceback)(:| occurred| detected| found| thrown| caught| message)\b', tool_content, re.IGNORECASE):
                    prev_tool, prev_file = _extract_tool_and_file(prev)
                    cur_tool, cur_file = _extract_tool_and_file(msg)
                    if prev_tool and cur_tool and prev_tool == cur_tool:
                        score -= 2
                    break

    # Context-aware: first occurrence of a tool pattern gets bonus
    if prev_msgs and role == "assistant" and msg.get("tool_calls"):
        for tc in msg.get("tool_calls") or []:
            if isinstance(tc, dict):
                tool = tc.get("function", {}).get("name", "")
                if tool:
                    seen = any(
                        tool in str(prev.get("tool_calls", ""))
                        for prev in prev_msgs[-max(1, window * 2):]
                    )
                    if not seen:
                        score += 1

    return max(0, min(10, score))


def classify(score: int, keep_thresh: int, drop_thresh: int) -> str:
    """Classify a score into HIGH / MEDIUM / LOW."""
    if score >= keep_thresh:
        return "HIGH"
    if score >= drop_thresh:
        return "MEDIUM"
    return "LOW"


def score_all_turns(
    messages: List[Dict[str, Any]],
    head_end: int,
    tail_start: int,
    keep_thresh: int = 7,
    drop_thresh: int = 3,
    retry_window: int = 3,
) -> List[tuple[int, str]]:
    """Score all turns in the compressible region.

    Returns list of (score, classification) for each turn between
    head_end and tail_start.
    """
    region = messages[head_end:tail_start]
    results = []

    for i, msg in enumerate(region):
        context_start = max(0, i - retry_window)
        prev = messages[head_end:head_end + context_start]
        score = score_turn(msg, prev_msgs=prev, turn_index=i, window=retry_window)
        classification = classify(score, keep_thresh, drop_thresh)
        results.append((score, classification))

    return results
