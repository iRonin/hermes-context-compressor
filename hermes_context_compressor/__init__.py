"""hermes-context-compressor — Smart context compression for Hermes Agent.

Heuristic turn scoring, retry loop detection, and configurable importance
thresholds for intelligent context management.

Usage as Hermes plugin:
    context:
      engine: "ironin-compressor"

    smart_compressor:
      keep_threshold: 7
      drop_threshold: 3
      max_retries_window: 3
      summarize_low: false
"""

from .config import parse_config, DEFAULTS
from .scorer import score_turn, classify, score_all_turns
from .pattern_detector import detect_retry_loops, collapse_retry_loops
from .compressor import IroninCompressor

__version__ = "0.1.0"
