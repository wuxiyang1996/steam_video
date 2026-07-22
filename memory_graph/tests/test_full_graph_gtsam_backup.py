from __future__ import annotations

import pytest

from memory_graph.types import CausalTemporalOverlay, MemoryNode, TimeSpan
from steam_video_new.implicit_world_model.full_graph_iwm import (
    ActionKind,
    CursorBeliefState,
    GTSAMMultiTrajectoryBeliefUpdater,
    LegalGraphAction,
    RetainedEvidenceGraph,
)
from factor_graph.gtsam_backend import GTSAM_AVAILABLE


def _node() -> MemoryNode:
    return MemoryNode(
        node_id="l1:one",
        video_id="video:gtsam-adapter",
        time_span=TimeSpan(0.0, 1.0),
        provenance={"producer": "test"},
        node_type="observation",
        text="person opens door",
    )


def _overlay(node: MemoryNode) -> CausalTemporalOverlay:
    return CausalTemporalOverlay(
        overlay_id="overlay:gtsam-adapter",
        example_id="example:gtsam-adapter",
        video_id=node.video_id,
        l1_observations=[node],
        atomic_events=[],
        relations=[],
    )


def test_gtsam_backup_fails_explicitly_when_dependency_is_unavailable(
    monkeypatch,
) -> None:
    node = _node()
    updater = GTSAMMultiTrajectoryBeliefUpdater(_overlay(node))
    import steam_video_new.implicit_world_model.full_graph_iwm.gtsam_backup as module

    monkeypatch.setattr(module, "GTSAM_AVAILABLE", False)
    previous = CursorBeliefState("belief:before", "question", remaining_reads=1)
    structural = CursorBeliefState(
        "belief:after",
        "question",
        current_node_id=node.node_id,
        acquired_evidence=(node.node_id,),
        remaining_reads=0,
        step=1,
    )
    action = LegalGraphAction(
        "action:read",
        ActionKind.START_AT,
        target_id=node.node_id,
        reads_evidence=True,
    )
    graph = RetainedEvidenceGraph(
        "graph:gtsam-adapter",
        (node,),
        (),
        (),
        1,
    )

    with pytest.raises(RuntimeError, match="GTSAM backup is unavailable"):
        updater.update_after_action(
            "trajectory:one",
            previous,
            structural,
            action,
            node,
            graph,
        )


@pytest.mark.skipif(
    not GTSAM_AVAILABLE, reason="GTSAM is unavailable in this interpreter"
)
def test_gtsam_backup_reuses_one_real_measurement_across_trajectories() -> None:
    node = _node()
    updater = GTSAMMultiTrajectoryBeliefUpdater(_overlay(node))
    previous = CursorBeliefState("belief:before", "question", remaining_reads=1)
    structural = CursorBeliefState(
        "belief:after",
        "question",
        current_node_id=node.node_id,
        acquired_evidence=(node.node_id,),
        remaining_reads=0,
        step=1,
    )
    action = LegalGraphAction(
        "action:read",
        ActionKind.START_AT,
        target_id=node.node_id,
        reads_evidence=True,
    )
    graph = RetainedEvidenceGraph("graph:gtsam-adapter", (node,), (), (), 1)

    first = updater.update_after_action(
        "trajectory:one", previous, structural, action, node, graph
    )
    second = updater.update_after_action(
        "trajectory:two", previous, structural, action, node, graph
    )

    assert first == second
    assert first.acquired_evidence == (node.node_id,)
    assert first.remaining_reads == 0
    assert updater.audit_records[-1]["shared_measurement_reused"] is True
