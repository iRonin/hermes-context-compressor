"""ironin-compressor plugin entry point for Hermes Agent.

This directory registers the IroninCompressor as a context engine plugin.
The core library lives in the hermes-context-compressor repo.
"""

from __future__ import annotations

import sys
import os

# Resolve path: this file is at plugins/context_engine/ironin_compressor/__init__.py
_plugin_dir = os.path.dirname(os.path.abspath(__file__))
_plugin_dir = os.path.realpath(_plugin_dir)
_repo_root = os.path.dirname(os.path.dirname(os.path.dirname(_plugin_dir)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from hermes_context_compressor.compressor import IroninCompressor

__version__ = "0.1.0"


def _load_smart_compressor_config() -> dict:
    """Read smart_compressor config from ~/.hermes/config.yaml."""
    try:
        import yaml
        config_path = os.path.expanduser("~/.hermes/config.yaml")
        if os.path.exists(config_path):
            with open(config_path) as f:
                cfg = yaml.safe_load(f) or {}
            return cfg.get("smart_compressor", {})
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
        return True
    except Exception:
        return False
