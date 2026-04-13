"""Config parsing and defaults for ironin-compressor."""

from __future__ import annotations

from typing import Any, Dict

DEFAULTS: Dict[str, Any] = {
    # Scoring thresholds (used when target_tokens is not set)
    "keep_threshold": 7,       # score >= this -> keep verbatim
    "drop_threshold": 4,       # score < this -> drop or brief mention
    # Pattern detection
    "max_retries_window": 3,   # same tool+file within N turns -> collapse
    # Behaviour
    "summarize_low": False,    # include LOW-value turns in LLM summary?
    "summarize_high": False,   # also summarize HIGH turns (for iterative compression safety)
    "preserve_tool_integrity": True,  # always maintain tool_call/result pairs
    # Performance
    "skip_scoring_under_msgs": 6,  # don't score if conversation is tiny
    # Adaptive compression (overrides fixed thresholds when set)
    "target_tokens": 0,        # target output token budget (0 = use fixed thresholds)
}

VALID_KEYS = frozenset(DEFAULTS.keys())

# Clamp ranges per key
_CLAMPS: Dict[str, tuple] = {
    "keep_threshold": (4, 10),
    "drop_threshold": (0, 6),
    "max_retries_window": (2, 10),
    "skip_scoring_under_msgs": (4, 20),
}


def parse_config(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Parse user config dict, applying clamps and defaults.

    Only keys from ``DEFAULTS`` are accepted. Unknown keys are silently
    ignored (future-proofing).
    """
    result = dict(DEFAULTS)
    for key, val in raw.items():
        if key not in VALID_KEYS:
            continue
        try:
            val = int(val) if key != "summarize_low" else bool(val)
        except (TypeError, ValueError):
            continue  # keep default on bad type
        if key in _CLAMPS:
            lo, hi = _CLAMPS[key]
            val = max(lo, min(hi, val))
        result[key] = val

    # Invariant: drop < keep always
    if result["drop_threshold"] >= result["keep_threshold"]:
        result["drop_threshold"] = result["keep_threshold"] - 1

    return result
