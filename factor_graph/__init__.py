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
from .post_read_verifier import PostReadCategoricalVerifier
from .navigation_backend import GTSAMExecutedReadBeliefBackend
from .correction_policy import (
    BackupTriggerDecision,
    CategoricalBackupTriggerPolicy,
    CorrectionMode,
)

__all__ = [
    "BeliefLabel",
    "BackupTriggerDecision",
    "CalibrationEntry",
    "FactorSpec",
    "GTSAM_AVAILABLE",
    "GTSAMDiscreteBeliefGraph",
    "GTSAMExecutedReadBeliefBackend",
    "GTSAMBeliefSession",
    "GTSAMOverlayAdapter",
    "InferenceResult",
    "MeasurementCalibrationRegistry",
    "CategoricalBackupTriggerPolicy",
    "CorrectionMode",
    "MeasurementJournal",
    "MeasurementOutcome",
    "OverlayInferenceResult",
    "OverlayParityResult",
    "PostReadCategoricalVerifier",
    "RelationMeasurement",
    "SessionUpdateResult",
    "VerifierDecision",
    "VariableSpec",
    "changed_variables",
    "measurement_from_execution",
    "project_probability",
]
