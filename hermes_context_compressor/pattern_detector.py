"""Pattern detector for ironin-compressor.

Detects retry loops (same tool + same file repeated) and collapses
them into a single entry so the summary doesn't waste tokens on
intermediate failed attempts.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

_FILE_PATH_RE = re.compile(r'["\']?path["\']?\s*[:=]\s*["\']([^"\']+)["\']')


def _extract_tool_file_pair(msg: Dict[str, Any]) -> Tuple[str, str]:
    """Extract (tool_name, file_path) from a message."""
    tool = ""
    path = ""

    for tc in msg.get("tool_calls") or []:
        if isinstance(tc, dict):
            fn = tc.get("function", {})
            tool = fn.get("name", "")
            args = fn.get("arguments", "")
            m = _FILE_PATH_RE.search(args)
            if m:
                path = m.group(1)

    if msg.get("role") == "tool":
        content = msg.get("content", "")
        m = re.search(r'(?:file|path)\s+["\']?([/\w.\-]+)["\']?', content)
        if m:
            path = m.group(1)

    return tool, path


def detect_retry_loops(
    messages: List[Dict[str, Any]],
    window: int = 3,
) -> List[Tuple[int, int, str]]:
    """Detect retry loops in a message sequence.

    A retry loop is when the same tool operates on the same file
    within ``window`` consecutive turns.

    Returns list of (start_idx, end_idx, summary_description).
    """
    if not messages:
        return []

    loops: List[Tuple[int, int, str]] = []
    n = len(messages)
    i = 0

    while i < n:
        tool, path = _extract_tool_file_pair(messages[i])
        if not tool:
            i += 1
            continue

        j = i + 1
        seen_same = False
        while j < min(i + window * 2, n):
            t2, p2 = _extract_tool_file_pair(messages[j])
            if t2 == tool and (not path or p2 == path):
                seen_same = True
                j += 1
            elif messages[j].get("role") == "tool":
                j += 1
            else:
                break

        if seen_same and (j - i) >= 3:
            first_msg = messages[i]
            content = first_msg.get("content", "")[:100]
            desc = f"{tool} loop on {path or 'unknown'} ({j - i} attempts)"
            loops.append((i, j, desc))
            i = j
        else:
            i += 1

    return loops


def collapse_retry_loops(
    messages: List[Dict[str, Any]],
    loops: List[Tuple[int, int, str]],
) -> List[Dict[str, Any]]:
    """Collapse detected retry loops into single representative messages."""
    if not loops:
        return messages

    result: List[Dict[str, Any]] = []
    loop_ranges = {start: (end, desc) for start, end, desc in loops}

    i = 0
    while i < len(messages):
        if i in loop_ranges:
            end, desc = loop_ranges[i]
            first = messages[i].copy()
            last = messages[end - 1].copy()

            first_content = first.get("content", "")
            first["content"] = f"[Retry loop detected: {desc}]\n{first_content}"

            last_content = last.get("content", "")
            last["content"] = f"[Final result after {desc}]\n{last_content}"

            result.append(first)
            if end - i > 2:
                result.append(last)
            else:
                for msg in messages[i:end]:
                    result.append(msg.copy())
            i = end
        else:
            result.append(messages[i].copy())
            i += 1

    return result
