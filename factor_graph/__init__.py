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
from .measurement import (
    CalibrationEntry,
    MeasurementCalibrationRegistry,
    MeasurementJournal,
    MeasurementOutcome,
    RelationMeasurement,
    VerifierDecision,
    measurement_from_execution,
)
from .session import GTSAMBeliefSession, SessionUpdateResult

__all__ = [
    "BeliefLabel",
    "CalibrationEntry",
    "FactorSpec",
    "GTSAM_AVAILABLE",
    "GTSAMDiscreteBeliefGraph",
    "GTSAMBeliefSession",
    "GTSAMOverlayAdapter",
    "InferenceResult",
    "MeasurementCalibrationRegistry",
    "MeasurementJournal",
    "MeasurementOutcome",
    "OverlayInferenceResult",
    "OverlayParityResult",
    "RelationMeasurement",
    "SessionUpdateResult",
    "VerifierDecision",
    "VariableSpec",
    "changed_variables",
    "measurement_from_execution",
    "project_probability",
]
