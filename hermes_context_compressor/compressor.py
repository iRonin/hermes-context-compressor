"""IroninCompressor — smart context compression for Hermes Agent.

Extends the built-in ContextCompressor with heuristic turn scoring,
retry loop detection, and configurable importance thresholds.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from agent.context_compressor import ContextCompressor, estimate_messages_tokens_rough

from .scorer import score_all_turns
from .pattern_detector import detect_retry_loops, collapse_retry_loops
from .config import parse_config, DEFAULTS

logger = logging.getLogger(__name__)

_LOW_TURNS_MARKER = "[{n} low-value turns omitted — iterative edits/retries with no substantive content]"


class IroninCompressor(ContextCompressor):
    """Smart context compressor with heuristic scoring."""

    @property
    def name(self) -> str:
        return "ironin-compressor"

    def __init__(
        self,
        model: str,
        threshold_percent: float = 0.50,
        protect_first_n: int = 3,
        protect_last_n: int = 20,
        summary_target_ratio: float = 0.20,
        summary_ratio: float = 0.20,
        quiet_mode: bool = False,
        summary_model_override: str = None,
        base_url: str = "",
        api_key: str = "",
        config_context_length: int | None = None,
        provider: str = "",
        api_mode: str = "",
        config: Dict[str, Any] | None = None,
    ):
        super().__init__(
            model=model,
            threshold_percent=threshold_percent,
            protect_first_n=protect_first_n,
            protect_last_n=protect_last_n,
            summary_target_ratio=summary_target_ratio,
            summary_ratio=summary_ratio,
            quiet_mode=quiet_mode,
            summary_model_override=summary_model_override,
            base_url=base_url,
            api_key=api_key,
            config_context_length=config_context_length,
            provider=provider,
            api_mode=api_mode,
        )

        self._sc_config = parse_config(config or {})
        self._keep_threshold = self._sc_config["keep_threshold"]
        self._drop_threshold = self._sc_config["drop_threshold"]
        self._retry_window = self._sc_config["max_retries_window"]
        self._summarize_low = self._sc_config["summarize_low"]
        self._summarize_high = self._sc_config["summarize_high"]
        self._skip_under = self._sc_config["skip_scoring_under_msgs"]
        self._last_compression_stats: Dict[str, Any] = {}

    def _estimate_output_tokens(
        self,
        region: List[Dict[str, Any]],
        scored: List[tuple[int, str]],
        keep_thresh: int,
        drop_thresh: int,
    ) -> tuple[int, int, int]:
        """Estimate output token budget for given thresholds.
        
        Returns (high_tokens, medium_summary_tokens, low_tokens).
        HIGH kept verbatim, MEDIUM summarized at ~20%, LOW dropped.
        """
        high_tokens = 0
        medium_tokens = 0
        low_tokens = 0
        for msg, (score, cls) in zip(region, scored):
            msg_tokens = estimate_messages_tokens_rough([msg])
            if cls == "HIGH":
                high_tokens += msg_tokens
            elif cls == "MEDIUM":
                medium_tokens += msg_tokens
            else:
                low_tokens += msg_tokens
        
        medium_summary = int(medium_tokens * 0.20)
        return high_tokens, medium_summary, low_tokens

    def _find_optimal_thresholds(
        self,
        region: List[Dict[str, Any]],
        raw_scores: List[int],
        target_tokens: int,
        summary_ratio: float = 0.20,
    ) -> tuple[int, int]:
        """Find keep/drop thresholds that hit the target token budget.
        
        Uses raw scores (0-10) and re-classifies at different thresholds
        to find the combination closest to target without going over.
        """
        # Try all valid threshold combinations
        best_keep = self._keep_threshold
        best_drop = self._drop_threshold
        best_diff = float('inf')
        
        from .scorer import classify
        
        for keep in range(6, 10):
            for drop in range(2, keep):
                high_tokens = 0
                medium_tokens = 0
                for msg, score in zip(region, raw_scores):
                    cls = classify(score, keep, drop)
                    msg_tokens = estimate_messages_tokens_rough([msg])
                    if cls == "HIGH":
                        high_tokens += msg_tokens
                    elif cls == "MEDIUM":
                        medium_tokens += msg_tokens
                
                output = high_tokens + int(medium_tokens * summary_ratio)
                diff = abs(output - target_tokens)
                
                # Prefer being under target but close to it
                if output <= target_tokens and diff < best_diff:
                    best_diff = diff
                    best_keep = keep
                    best_drop = drop
        
        return best_keep, best_drop

    def compress(
        self,
        messages: List[Dict[str, Any]],
        current_tokens: int = None,
        focus_topic: str = None,
        target_tokens: int = 0,
    ) -> List[Dict[str, Any]]:
        """Compress using smart scoring instead of blanket summarization.
        
        Args:
            target_tokens: Optional target output token budget. When set (>0),
                dynamically adjusts keep/drop thresholds to hit this target.
                Useful for /compress <target> CLI commands.
                When 0 (default), uses configured keep/drop thresholds.
        """
        n_messages = len(messages)
        _min_for_compress = self.protect_first_n + 3 + 1
        if n_messages <= _min_for_compress:
            return messages

        display_tokens = current_tokens or self.last_prompt_tokens or estimate_messages_tokens_rough(messages)

        # Phase 1: Prune old tool results
        messages, pruned_count = self._prune_old_tool_results(
            messages, protect_tail_count=self.protect_last_n,
            protect_tail_tokens=self.tail_token_budget,
        )
        if pruned_count and not self.quiet_mode:
            logger.info("Pre-compression: pruned %d old tool result(s)", pruned_count)

        # Phase 2: Determine boundaries
        compress_start = self.protect_first_n
        compress_start = self._align_boundary_forward(messages, compress_start)
        compress_end = self._find_tail_cut_by_tokens(messages, compress_start)

        if compress_start >= compress_end:
            return messages

        region = list(messages[compress_start:compress_end])
        n_region = len(region)

        # Phase 2b: Detect and collapse retry loops
        loops = detect_retry_loops(region, window=self._retry_window)
        if loops:
            region = collapse_retry_loops(region, loops)
            if not self.quiet_mode:
                logger.info("Collapsed %d retry loop(s)", len(loops))

        # Phase 3: Score turns
        # Always score first with raw scores, then classify
        if n_region >= self._skip_under:
            raw_scored = []
            from .scorer import score_turn, classify
            for i, msg in enumerate(region):
                context_start = max(0, i - self._retry_window)
                prev = region[max(0, context_start):i]
                score = score_turn(msg, prev_msgs=prev, turn_index=i, window=self._retry_window)
                raw_scored.append(score)
        else:
            raw_scored = [5] * len(region)
        
        # Adaptive threshold selection if target is passed
        keep_thresh = self._keep_threshold
        drop_thresh = self._drop_threshold
        
        if target_tokens > 0:
            keep_thresh, drop_thresh = self._find_optimal_thresholds(
                region, raw_scored, target_tokens
            )
            if not self.quiet_mode:
                logger.info(
                    "Adaptive thresholds for target %d tokens: keep=%d, drop=%d",
                    target_tokens, keep_thresh, drop_thresh,
                )
        
        # Classify with chosen thresholds
        from .scorer import classify as _classify
        scored = [(s, _classify(s, keep_thresh, drop_thresh)) for s in raw_scored]

        # Phase 4: Partition by classification
        high_turns: List[int] = []
        medium_turns: List[int] = []
        low_count = 0

        for idx, (score, classification) in enumerate(scored):
            if classification == "HIGH":
                high_turns.append(idx)
            elif classification == "MEDIUM":
                medium_turns.append(idx)
            else:
                low_count += 1

        if not self.quiet_mode:
            logger.info(
                "Scored %d turns: %d HIGH (verbatim), %d MEDIUM (summary), %d LOW (dropped)",
                n_region, len(high_turns), len(medium_turns), low_count,
            )

        # Phase 5: If all HIGH, no compression benefit
        if not medium_turns and not low_count:
            if not self.quiet_mode:
                logger.info("All turns scored HIGH — no compression benefit, returning as-is")
            return messages

        # Phase 6: Build the middle section
        kept_msgs = [region[i].copy() for i in high_turns]

        # Optional: Generate a summary of HIGH turns for iterative compression safety
        # When summarize_high=True, we keep HIGH verbatim BUT also produce a summary
        # so that if the session compresses again, the essence of these turns survives
        high_summary = None
        if self._summarize_high and high_turns:
            high_msgs_for_summary = [region[i] for i in high_turns]
            high_summary = self._generate_summary(high_msgs_for_summary, focus_topic=focus_topic)
            if high_summary and not self.quiet_mode:
                logger.info("Generated HIGH-turn summary (%d turns summarized for safety)", len(high_turns))

        summary = None
        if medium_turns:
            medium_msgs = [region[i] for i in medium_turns]
            if self._summarize_low and low_count > 0:
                low_indices = [i for i, (_, c) in enumerate(scored) if c == "LOW"]
                medium_msgs = [region[i] for i in medium_turns + low_indices]
            summary = self._generate_summary(medium_msgs, focus_topic=focus_topic)

        # Phase 7: Assemble compressed message list
        compressed = []

        # Head
        for i in range(compress_start):
            msg = messages[i].copy()
            if i == 0 and msg.get("role") == "system" and self.compression_count == 0:
                msg["content"] = (
                    (msg.get("content") or "")
                    + "\n\n[Note: Some earlier conversation turns have been compacted using smart scoring. High-value turns (decisions, fixes, key outputs) are preserved verbatim. Low-value iterative edits were omitted.]"
                )
            compressed.append(msg)

        # LOW turns marker
        if low_count > 0:
            compressed.append({
                "role": "user",
                "content": _LOW_TURNS_MARKER.format(n=low_count),
            })

        # HIGH turns: optional summary for iterative compression safety
        if high_summary:
            compressed.append({
                "role": "user",
                "content": "[HIGH-VALUE TURNS SUMMARY — The full messages follow verbatim. This summary exists for safety if the session is compressed again later.]\n" + high_summary,
            })

        # HIGH turns verbatim
        compressed.extend(kept_msgs)

        # LLM summary of MEDIUM turns
        if summary:
            compressed.append({"role": "user", "content": summary})
        elif medium_turns and not summary:
            compressed.append({
                "role": "user",
                "content": f"[{len(medium_turns)} conversation turns summarized but summary generation failed — see recent messages below]",
            })

        # Tail
        for i in range(compress_end, n_messages):
            compressed.append(messages[i].copy())

        compressed = self._sanitize_tool_pairs(compressed)

        new_estimate = estimate_messages_tokens_rough(compressed)
        saved = display_tokens - new_estimate
        self._last_compression_stats = {
            "total_turns": n_region,
            "high_kept": len(high_turns),
            "medium_summarized": len(medium_turns),
            "low_dropped": low_count,
            "tokens_saved_estimate": saved,
            "target_tokens": target_tokens if target_tokens > 0 else None,
            "actual_keep_threshold": keep_thresh,
            "actual_drop_threshold": drop_thresh,
        }

        if not self.quiet_mode:
            adaptive_note = ""
            if target_tokens > 0:
                adaptive_note = f" (adaptive: keep={keep_thresh}, drop={drop_thresh})"
            logger.info(
                "Smart compressed: %d -> %d messages (%d HIGH kept, %d MEDIUM summarized, %d LOW dropped, ~%d tokens saved)%s",
                n_messages, len(compressed),
                len(high_turns), len(medium_turns), low_count, saved,
                adaptive_note,
            )
            logger.info("Compression #%d complete", self.compression_count + 1)

        self.compression_count += 1
        return compressed

    def get_status(self) -> Dict[str, Any]:
        """Return status with smart-compressor stats."""
        status = super().get_status()
        if self._last_compression_stats:
            status.update(self._last_compression_stats)
        return status
