"""Clean v2 architecture for world-model-guided multi-path reasoning."""

from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "ReasoningRuntime",
    "belief",
    "evaluation",
    "evidence",
    "navigation",
    "planner",
    "qformer",
    "world_model",
]


def __getattr__(name: str) -> Any:
    """Keep independent subpackages isolated from optional runtime dependencies."""

    if name == "ReasoningRuntime":
        from .runtime import ReasoningRuntime

        return ReasoningRuntime
    if name in {"belief", "evaluation", "evidence", "navigation", "planner", "qformer", "world_model"}:
        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(name)
