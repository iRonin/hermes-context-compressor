"""ironin-compressor plugin entry point for Hermes Agent.

This directory registers the IroninCompressor as a context engine plugin.
The core library lives in the hermes-context-compressor repo.
"""

from __future__ import annotations

import sys
import os

# Resolve path: this file is at plugins/context_engine/ironin_compressor/__init__.py
# The hermes_context_compressor package is at the repo root
_plugin_dir = os.path.dirname(os.path.abspath(__file__))
# Follow symlinks to find the real location
_plugin_dir = os.path.realpath(_plugin_dir)
# plugins/context_engine/ironin_compressor -> go up 3 levels to repo root
_repo_root = os.path.dirname(os.path.dirname(os.path.dirname(_plugin_dir)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from hermes_context_compressor.compressor import IroninCompressor

__version__ = "0.1.0"


def register(ctx) -> None:
    """Register the ironin-compressor context engine."""
    ctx.register_context_engine(IroninCompressor())


def is_available() -> bool:
    """Lightweight availability check."""
    try:
        return True
    except Exception:
        return False
