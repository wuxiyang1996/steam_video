"""Compatibility continuous-belief boundary.

The canonical GTSAM design and binding-status discussion live in
``steam_video/factor_graph``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from memory_graph.types import CausalTemporalOverlay


@dataclass(frozen=True)
class ContinuousBeliefSummary:
    """Opaque summary projected into the discrete backend reference."""

    backend_name: str
    backend_ref: str
    latent_variable_count: int
    measurement_count: int


class ContinuousBeliefSmoother(Protocol):
    """Interface a GTSAM backend may implement when continuous latents exist."""

    name: str

    def update(
        self,
        overlay: CausalTemporalOverlay,
        acquired_evidence: frozenset[str],
    ) -> ContinuousBeliefSummary: ...


class ObservedIntervalSmoother:
    """Current no-optimization implementation for observed event intervals."""

    name = "observed_intervals/v0.1"

    def update(
        self,
        overlay: CausalTemporalOverlay,
        acquired_evidence: frozenset[str],
    ) -> ContinuousBeliefSummary:
        event_ids = {node.node_id for node in overlay.atomic_events}
        measurements = len(event_ids & acquired_evidence)
        return ContinuousBeliefSummary(
            backend_name=self.name,
            backend_ref=f"observed:{measurements}",
            latent_variable_count=0,
            measurement_count=measurements,
        )
