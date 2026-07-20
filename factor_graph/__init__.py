"""Canonical factor-graph subsystem for exploration-time belief."""

from .categorical import BeliefLabel, project_probability
from .gtsam_backend import (
    GTSAM_AVAILABLE,
    FactorSpec,
    GTSAMDiscreteBeliefGraph,
    InferenceResult,
    VariableSpec,
)
from .overlay_adapter import (
    GTSAMOverlayAdapter,
    OverlayInferenceResult,
    OverlayParityResult,
    changed_variables,
)

__all__ = [
    "BeliefLabel",
    "FactorSpec",
    "GTSAM_AVAILABLE",
    "GTSAMDiscreteBeliefGraph",
    "GTSAMOverlayAdapter",
    "InferenceResult",
    "OverlayInferenceResult",
    "OverlayParityResult",
    "VariableSpec",
    "changed_variables",
    "project_probability",
]
