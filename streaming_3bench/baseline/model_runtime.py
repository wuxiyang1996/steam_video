"""Shared model-loading options for local Qwen evaluation."""

from __future__ import annotations

import os
from typing import Any


def attention_kwargs() -> dict[str, Any]:
    """Return the requested Transformers attention backend, if configured."""
    implementation = os.environ.get("QWEN_ATTN_IMPLEMENTATION", "").strip()
    if not implementation or implementation.lower() in {"auto", "default", "none"}:
        return {}
    return {"attn_implementation": implementation}
