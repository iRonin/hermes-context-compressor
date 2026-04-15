"""ironin-compressor plugin entry point for Hermes Agent.

This directory registers the IroninCompressor as a context engine plugin.
The core library lives in the hermes-context-compressor repo.
"""

from __future__ import annotations

from hermes_context_compressor.compressor import IroninCompressor

__version__ = "0.1.0"


def _load_smart_compressor_config() -> dict:
    """Read smart_compressor config from the active Hermes profile."""
    try:
        from hermes_cli.config import load_config
        cfg = load_config()
        return cfg.get("smart_compressor", {}) or {}
    except Exception:
        pass
    return {}


def register(ctx) -> None:
    """Register the ironin-compressor context engine."""
    config = _load_smart_compressor_config()
    ctx.register_context_engine(IroninCompressor(config=config))


def is_available() -> bool:
    """Lightweight availability check."""
    try:
        from hermes_context_compressor.compressor import IroninCompressor
        return True
    except ImportError:
        return False
