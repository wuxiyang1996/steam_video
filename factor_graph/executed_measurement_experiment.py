"""Phase C executed-read measurement pilot with one real Video_Skills replay."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
from typing import Any

from memory_graph.navigation import GraphReadAction, NavigationActionType
from memory_graph.types import (
    CausalTemporalOverlay,
    MemoryNode,
    RelationBelief,
    RelationStatus,
    TimeSpan,
)
from steam_video_new.implicit_world_model.l15_graph_navigator import (
    FactorizedBeliefBackend,
    VideoSkillsL2Adapter,
    load_overlay_artifact,
)
from steam_video_new.implicit_world_model.l15_graph_navigator.contracts import (
    GraphReadExecution,
)

from .measurement import (
    MeasurementCalibrationRegistry,
    MeasurementJournal,
    MeasurementOutcome,
    VerifierDecision,
)
from .overlay_adapter import GTSAMOverlayAdapter, changed_variables
from .session import GTSAMBeliefSession


def run_experiment(overlay_path: Path) -> dict[str, Any]:
    calibration = MeasurementCalibrationRegistry.pilot()
    real = _run_real_video_skills_replay(overlay_path, calibration)
    controlled = _run_controlled_coupled_pilot(calibration)
    gates = {
        "real_video_skills_read_executed": real["execution_status"] == "executed",
        "real_read_has_grounded_l1_evidence": real["grounded_evidence_ref_count"] > 0,
        "real_measurement_changes_direct_target": real["direct_changed_count"] > 0,
        "query_is_not_a_factor": real["factor_count_before_verifier"]
        == real["factor_count_after_read_before_verifier"],
        "measurement_is_idempotent": controlled["idempotent_replay"],
        "coupled_measurement_propagates": controlled["propagated_changed_count"] > 0,
        "no_loop_blocks_propagation": controlled["no_loop_propagated_changed_count"] == 0,
        "shuffled_target_changes_result": controlled["shuffled_disagreement_count"] > 0,
        "conflicting_measurements_are_retained": controlled["journal_size_after_conflict"]
        == 2,
    }
    return {
        "schema_version": "gtsam_executed_measurement_experiment/v0.1",
        "status": "phase_c_integration_pilot_not_calibration_benchmark",
        "runtime": {
            "gtsam": version("gtsam"),
            "python": platform.python_version(),
            "video_skills_runtime": True,
            "llm_numeric_output": False,
            "calibration_version": calibration.version,
        },
        "gates": gates,
        "all_gates_pass": all(gates.values()),
        "real_video_skills_replay": real,
        "controlled_coupled_pilot": controlled,
        "interpretation": (
            "The real arm exercises Video_Skills retrieval and grounded measurement "
            "construction using a persisted categorical verifier decision. The coupled "
            "arm tests propagation mechanics; neither arm constitutes calibrated accuracy."
        ),
    }


def _run_real_video_skills_replay(
    path: Path,
    calibration: MeasurementCalibrationRegistry,
) -> dict[str, Any]:
    loaded = load_overlay_artifact(path)
    overlay = loaded.overlay
    edge, relation, verifier = _select_replay_relation(overlay)
    action = GraphReadAction(
        _action_type(relation),
        source_id=edge.src,
        target_ids=(edge.dst,),
        relation=relation,
        rationale="phase-c executed measurement replay",
    )
    belief = FactorizedBeliefBackend().initialize(
        "Verify the selected relation after a real graph read.",
        overlay,
        seed_evidence=(edge.src,),
        missing_roles=("verification",),
    )
    session = GTSAMBeliefSession(
        overlay,
        calibration=calibration,
        activate_persisted_verified_measurements=False,
    )
    before_read = session.snapshot()
    execution = VideoSkillsL2Adapter(use_video_skills_runtime=True).execute(
        belief, action, overlay
    )
    after_read_before_verifier = session.snapshot()
    evidence_refs = tuple(
        str(value) for value in execution.skill_invocation.get("evidence_refs") or []
    )
    decision = VerifierDecision(
        edge_id=edge.edge_id,
        relation=relation,
        outcome=MeasurementOutcome.SUPPORTS,
        verifier_name="persisted_hard_verifier_replay",
        verifier_version=str(
            overlay.metadata.get("hard_verifier_protocol") or "overlay-provenance/v1"
        ),
        evidence_refs=evidence_refs,
        reasons=tuple(str(value) for value in verifier.get("reasons") or []),
    )
    update = session.update(action=action, execution=execution, decision=decision)
    variable = update.measurement.variable_id
    return {
        "artifact": str(path),
        "artifact_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "overlay_id": overlay.overlay_id,
        "edge_id": edge.edge_id,
        "relation": relation,
        "relation_calibration_status": edge.status.value,
        "execution_status": execution.skill_invocation.get("status"),
        "skill_id": execution.skill_invocation.get("skill_id"),
        "real_observation_ids": [node.node_id for node in execution.observations],
        "grounded_evidence_ref_count": len(evidence_refs),
        "measurement_outcome": update.measurement.outcome.value,
        "measurement_id": update.measurement.measurement_id,
        "belief_before": before_read.categorical[variable].value,
        "belief_after_read_before_verifier": after_read_before_verifier.categorical[
            variable
        ].value,
        "belief_after_measurement": update.belief_after.categorical[variable].value,
        "factor_count_before_verifier": before_read.factor_count,
        "factor_count_after_read_before_verifier": after_read_before_verifier.factor_count,
        "factor_count_after_measurement": update.belief_after.factor_count,
        "direct_changed_count": len(update.direct_changed_variables),
        "propagated_changed_count": len(update.propagated_changed_variables),
        "journal_size": len(session.journal.measurements),
    }


def _run_controlled_coupled_pilot(
    calibration: MeasurementCalibrationRegistry,
) -> dict[str, Any]:
    overlay, action, execution, supports = _controlled_fixture()
    session = GTSAMBeliefSession(
        overlay,
        calibration=calibration,
        activate_persisted_verified_measurements=False,
    )
    frozen = session.snapshot()
    update = session.update(action=action, execution=execution, decision=supports)
    repeated = session.update(action=action, execution=execution, decision=supports)
    no_loop_before = session.adapter.infer(
        overlay,
        activate_verified_measurements=False,
        include_pairwise_factors=False,
        measurement_journal=MeasurementJournal(),
        calibration=calibration,
    )
    no_loop = session.snapshot(include_pairwise_factors=False)
    no_loop_changed = set(changed_variables(no_loop_before, no_loop))

    shuffled_journal = MeasurementJournal()
    state_variable = "relation::edge:state::state_transition"
    shuffled_journal.append(
        replace(
            update.measurement,
            measurement_id=f"{update.measurement.measurement_id}:shuffled",
            edge_id="edge:state",
            relation="state_transition",
            variable_id=state_variable,
        )
    )
    shuffled = GTSAMOverlayAdapter().infer(
        overlay,
        activate_verified_measurements=False,
        measurement_journal=shuffled_journal,
        calibration=calibration,
    )
    shuffled_disagreement = sum(
        update.belief_after.categorical[name] is not shuffled.categorical[name]
        for name in shuffled.categorical
    )

    reject_execution = GraphReadExecution(
        execution.observations,
        {**execution.skill_invocation, "node_id": "skill:track:a-b:counterevidence"},
    )
    rejects = replace(
        supports,
        outcome=MeasurementOutcome.REJECTS,
        verifier_name="identity_counterevidence_verifier",
        reasons=("controlled conflicting evidence",),
    )
    conflict_update = session.update(
        action=action,
        execution=reject_execution,
        decision=rejects,
    )
    target = update.measurement.variable_id
    return {
        "measurement_target": target,
        "belief_before": frozen.categorical[target].value,
        "belief_after": update.belief_after.categorical[target].value,
        "direct_changed_count": len(update.direct_changed_variables),
        "propagated_changed_count": len(update.propagated_changed_variables),
        "idempotent_replay": not repeated.appended and not repeated.changed_variables,
        "no_loop_direct_changed_count": len(no_loop_changed & {target}),
        "no_loop_propagated_changed_count": len(no_loop_changed - {target}),
        "shuffled_disagreement_count": shuffled_disagreement,
        "journal_size_after_conflict": len(session.journal.measurements),
        "belief_after_conflict": conflict_update.belief_after.categorical[target].value,
    }


def _controlled_fixture() -> tuple[
    CausalTemporalOverlay,
    GraphReadAction,
    GraphReadExecution,
    VerifierDecision,
]:
    l1_a = _node("l1:a", "observation", 0.0)
    l1_b = _node("l1:b", "observation", 2.0)
    event_a = _node("event:a", "atomic_event", 0.0, (l1_a.node_id,))
    event_b = _node("event:b", "atomic_event", 2.0, (l1_b.node_id,))
    overlay = CausalTemporalOverlay(
        overlay_id="overlay:phase-c-controlled",
        example_id="example:phase-c-controlled",
        video_id="video:phase-c-controlled",
        l1_observations=[l1_a, l1_b],
        atomic_events=[event_a, event_b],
        relations=[
            RelationBelief(
                "edge:identity",
                event_a.node_id,
                event_b.node_id,
                {"same_entity": 0.2},
                RelationStatus.UNCALIBRATED_PRIOR,
                0.5,
            ),
            RelationBelief(
                "edge:state",
                event_a.node_id,
                event_b.node_id,
                {"state_transition": 0.55},
                RelationStatus.UNCALIBRATED_PRIOR,
                0.5,
            ),
        ],
    )
    action = GraphReadAction(
        NavigationActionType.TRACK_ENTITY,
        event_a.node_id,
        (event_b.node_id,),
        "same_entity",
        "controlled identity inspection",
    )
    execution = GraphReadExecution(
        (event_b,),
        {
            "node_id": "skill:track:a-b",
            "skill_id": "retrieve_by_relation",
            "status": "executed",
            "outputs": {"real_observation_ids": [event_b.node_id]},
            "evidence_refs": [l1_b.node_id],
        },
    )
    decision = VerifierDecision(
        "edge:identity",
        "same_entity",
        MeasurementOutcome.SUPPORTS,
        "controlled_identity_verifier",
        "v1",
        (l1_b.node_id,),
    )
    return overlay, action, execution, decision


def _node(
    node_id: str,
    node_type: str,
    start_s: float,
    source_segments: tuple[str, ...] = (),
) -> MemoryNode:
    return MemoryNode(
        node_id=node_id,
        video_id="video:phase-c-controlled",
        time_span=TimeSpan(start_s, start_s + 1.0),
        provenance={"producer": "phase_c_controlled_fixture"},
        node_type=node_type,
        source_segments=list(source_segments),
    )


def _select_replay_relation(
    overlay: CausalTemporalOverlay,
) -> tuple[RelationBelief, str, dict[str, Any]]:
    candidates = []
    for edge in overlay.relations + overlay.l1_structural_relations:
        hard = edge.provenance.get("hard_verifier") or {}
        for relation, result in hard.items():
            if (
                isinstance(result, dict)
                and result.get("passed") is True
                and relation in edge.relation_probabilities
                and edge.relation_probabilities[relation] < 2.0 / 3.0
                and relation in _ACTION_TYPES
            ):
                candidates.append((edge, relation, result))
    if not candidates:
        raise ValueError("overlay has no eligible low-prior verified relation replay")
    return min(candidates, key=lambda item: (item[0].edge_id, item[1]))


_ACTION_TYPES = {
    "same_entity": NavigationActionType.TRACK_ENTITY,
    "same_object": NavigationActionType.TRACK_ENTITY,
    "same_instance_candidate": NavigationActionType.TRACK_ENTITY,
    "reappears_candidate": NavigationActionType.TRACK_ENTITY,
    "state_transition": NavigationActionType.INSPECT_STATE_CHANGE,
    "transition_support": NavigationActionType.FOLLOW_DEPENDENCY,
    "observation_support": NavigationActionType.FOLLOW_DEPENDENCY,
    "explains": NavigationActionType.CANDIDATE_CAUSE,
    "enables": NavigationActionType.CANDIDATE_CAUSE,
}


def _action_type(relation: str) -> NavigationActionType:
    return _ACTION_TYPES[relation]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overlay", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = run_experiment(args.overlay)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
