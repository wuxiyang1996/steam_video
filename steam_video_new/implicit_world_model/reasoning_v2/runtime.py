"""Small façade composing the separated reasoning-v2 components."""

from __future__ import annotations

from dataclasses import dataclass, replace

from steam_video_new.implicit_world_model.full_graph_iwm.contracts import (
    RetainedEvidenceGraph,
)

from .evidence import EvidenceMemory
from .evidence import from_retained_graph as evidence_from_legacy
from .navigation import NavigationGraph
from .navigation import from_retained_graph as navigation_from_legacy
from .planner import (
    PersistentMultiPathPlanner,
    PlanningDecision,
    ReasoningPathForest,
    apply_real_read,
)
from .world_model import RealEffectCorrector


@dataclass(frozen=True)
class ReasoningRuntime:
    memory: EvidenceMemory
    navigation: NavigationGraph
    planner: PersistentMultiPathPlanner
    forest: ReasoningPathForest

    @classmethod
    def from_legacy_graph(
        cls,
        graph: RetainedEvidenceGraph,
        *,
        entry_node_ids: tuple[str, ...],
        planner: PersistentMultiPathPlanner,
        forest: ReasoningPathForest,
    ) -> "ReasoningRuntime":
        return cls(
            memory=evidence_from_legacy(graph),
            navigation=navigation_from_legacy(graph, entry_node_ids=entry_node_ids),
            planner=planner,
            forest=forest,
        )

    def plan_next(self) -> PlanningDecision:
        return self.planner.plan(self.forest, self.memory, self.navigation)

    def execute(
        self,
        decision: PlanningDecision,
        *,
        corrector: RealEffectCorrector | None = None,
    ) -> "ReasoningRuntime":
        return replace(
            self,
            forest=apply_real_read(
                self.forest,
                decision,
                self.memory,
                corrector=corrector,
            ),
        )
