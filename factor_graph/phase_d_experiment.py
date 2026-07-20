"""Phase D grounded post-read verifier and coupled GTSAM experiment."""

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
from memory_graph.types import CausalTemporalOverlay, RelationBelief
from steam_video_new.implicit_world_model.l15_graph_navigator import (
    FactorizedBeliefBackend,
    VideoSkillsL2Adapter,
    load_overlay_artifact,
)

from .measurement import MeasurementCalibrationRegistry, MeasurementJournal
from .overlay_adapter import GTSAMOverlayAdapter, changed_variables
from .post_read_verifier import PostReadCategoricalVerifier
from .session import GTSAMBeliefSession


_ACTIONS = {
    "same_entity": NavigationActionType.TRACK_ENTITY,
    "state_transition": NavigationActionType.INSPECT_STATE_CHANGE,
    "transition_support": NavigationActionType.FOLLOW_DEPENDENCY,
    "contradicts": NavigationActionType.FOLLOW_DEPENDENCY,
}


def run_experiment(overlay_path: Path) -> dict[str, Any]:
    overlay = load_overlay_artifact(
        overlay_path,
        require_embedding_files=True,
        verify_embedding_checksums=True,
    ).overlay
    calibration = MeasurementCalibrationRegistry.pilot()
    runtime = VideoSkillsL2Adapter(use_video_skills_runtime=True)
    verifier = PostReadCategoricalVerifier()
    edges = {_relation(edge): edge for edge in overlay.relations}

    state_action, state_execution, state_decision = _read_and_verify(
        overlay, edges["state_transition"], runtime, verifier
    )
    state_session = _session(overlay, calibration)
    frozen = state_session.snapshot()
    state_update = state_session.update(
        action=state_action,
        execution=state_execution,
        decision=state_decision,
        previously_grounded_evidence_refs=tuple(
            _node(overlay, edges["state_transition"].src).source_segments
        ),
    )

    no_loop_before = GTSAMOverlayAdapter().infer(
        overlay,
        activate_verified_measurements=False,
        include_pairwise_factors=False,
        measurement_journal=MeasurementJournal(),
        calibration=calibration,
    )
    no_loop_after = GTSAMOverlayAdapter().infer(
        overlay,
        activate_verified_measurements=False,
        include_pairwise_factors=False,
        measurement_journal=state_session.journal,
        calibration=calibration,
    )
    no_loop_changed = set(changed_variables(no_loop_before, no_loop_after))

    shuffled_journal = MeasurementJournal()
    shuffled_target = edges["same_entity"]
    shuffled_journal.append(
        replace(
            state_update.measurement,
            measurement_id=f"{state_update.measurement.measurement_id}:shuffled",
            edge_id=shuffled_target.edge_id,
            relation="same_entity",
            variable_id=_variable(shuffled_target, "same_entity"),
        )
    )
    shuffled = GTSAMOverlayAdapter().infer(
        overlay,
        activate_verified_measurements=False,
        measurement_journal=shuffled_journal,
        calibration=calibration,
    )

    conflict_action, conflict_execution, conflict_decision = _read_and_verify(
        overlay, edges["contradicts"], runtime, verifier
    )
    conflict_update = state_session.update(
        action=conflict_action,
        execution=conflict_execution,
        decision=conflict_decision,
        previously_grounded_evidence_refs=tuple(
            _node(overlay, edges["contradicts"].src).source_segments
        ),
    )

    dependency_action, dependency_execution, dependency_decision = _read_and_verify(
        overlay, edges["transition_support"], runtime, verifier
    )
    dependency_session = _session(overlay, calibration)
    dependency_update = dependency_session.update(
        action=dependency_action,
        execution=dependency_execution,
        decision=dependency_decision,
        previously_grounded_evidence_refs=tuple(
            _node(overlay, edges["transition_support"].src).source_segments
        ),
    )

    state_variable = _variable(edges["state_transition"], "state_transition")
    identity_variable = _variable(edges["same_entity"], "same_entity")
    dependency_variable = _variable(edges["transition_support"], "transition_support")
    conflict_variable = _variable(edges["contradicts"], "contradicts")
    normal_labels = _labels(state_update.belief_after)
    shuffled_labels = _labels(shuffled)
    shuffled_disagreement = sorted(
        name
        for name in normal_labels
        if normal_labels[name] != shuffled_labels[name]
    )
    embedding_refs = [
        node.embedding_ref
        for node in overlay.atomic_events
        if node.embedding_ref is not None
    ]
    gates = {
        "all_real_graph_reads_executed": all(
            execution.skill_invocation.get("status") == "executed"
            for execution in (
                state_execution,
                conflict_execution,
                dependency_execution,
            )
        ),
        "post_read_verifiers_support": all(
            decision.outcome.value == "supports"
            for decision in (state_decision, conflict_decision, dependency_decision)
        ),
        "state_measurement_changes_direct_target": (
            state_variable in state_update.direct_changed_variables
        ),
        "state_measurement_propagates_to_identity": (
            identity_variable in state_update.propagated_changed_variables
        ),
        "frozen_has_no_measurement": not frozen.verified_measurement_targets,
        "no_loop_blocks_propagation": not (no_loop_changed - {state_variable}),
        "shuffled_target_changes_result": bool(shuffled_disagreement),
        "conflicting_measurements_are_retained": len(state_session.journal.measurements)
        == 2,
        "grounded_conflicting_measurement_is_retained": any(
            measurement.variable_id == conflict_variable
            for measurement in state_session.journal.measurements
        ),
        "verified_dependency_is_nonempty": (
            dependency_variable in dependency_update.direct_changed_variables
        ),
        "qwen_embedding_refs_are_retained": bool(embedding_refs)
        and all(ref.model == "Qwen/Qwen3-VL-Embedding-2B" for ref in embedding_refs),
    }
    return {
        "schema_version": "gtsam_phase_d_grounded_experiment/v0.1",
        "status": "derived_grounded_integration_smoke_not_calibration_benchmark",
        "production_ready": False,
        "runtime": {
            "gtsam": version("gtsam"),
            "python": platform.python_version(),
            "video_skills_runtime": True,
            "post_read_verifier": verifier.version,
            "calibration_version": calibration.version,
            "llm_numeric_output": False,
        },
        "source": {
            "artifact": str(overlay_path),
            "artifact_sha256": hashlib.sha256(overlay_path.read_bytes()).hexdigest(),
            "overlay_id": overlay.overlay_id,
            "derived_from": overlay.metadata.get("source_artifact"),
            "embedding_models": sorted({ref.model for ref in embedding_refs}),
            "embedding_ref_count": len(embedding_refs),
        },
        "gates": gates,
        "all_gates_pass": all(gates.values()),
        "arms": {
            "frozen": {"labels": _labels(frozen), "measurement_count": 0},
            "normal": {
                "decision": state_decision.to_dict(),
                "labels": normal_labels,
                "direct_changed": list(state_update.direct_changed_variables),
                "propagated_changed": list(state_update.propagated_changed_variables),
            },
            "no_loop": {
                "labels": _labels(no_loop_after),
                "changed": sorted(no_loop_changed),
            },
            "shuffled": {
                "labels": shuffled_labels,
                "disagreement_with_normal": shuffled_disagreement,
            },
            "conflicting": {
                "decision_outcome": conflict_decision.outcome.value,
                "labels": _labels(conflict_update.belief_after),
                "journal": state_session.journal.to_records(),
            },
            "verified_dependency": {
                "decision_outcome": dependency_decision.outcome.value,
                "labels": _labels(dependency_update.belief_after),
                "direct_changed": list(dependency_update.direct_changed_variables),
            },
        },
        "limitations": [
            "The accepted track remap is a transparent integration fixture, not an independent human gold label.",
            "Pilot likelihoods exercise inference mechanics and are not calibrated accuracy estimates.",
            "Equal-strength state and contradiction support may leave categorical labels unchanged; both measurements remain auditable.",
            "Production still requires a fixed, independently annotated multi-video case set.",
        ],
    }


def _read_and_verify(
    overlay: CausalTemporalOverlay,
    edge: RelationBelief,
    runtime: VideoSkillsL2Adapter,
    verifier: PostReadCategoricalVerifier,
):
    relation = _relation(edge)
    action = GraphReadAction(
        _ACTIONS[relation],
        source_id=edge.src,
        target_ids=(edge.dst,),
        relation=relation,
        rationale="phase-d grounded post-read verification",
    )
    belief = FactorizedBeliefBackend().initialize(
        "Inspect the grounded relation after a real graph read.",
        overlay,
        seed_evidence=(edge.src,),
        missing_roles=("verification",),
    )
    execution = runtime.execute(belief, action, overlay)
    decision = verifier.verify(
        action=action,
        execution=execution,
        overlay=overlay,
        edge_id=edge.edge_id,
        acquired_observation_ids=belief.acquired_evidence,
    )
    return action, execution, decision


def _session(overlay, calibration):
    return GTSAMBeliefSession(
        overlay,
        calibration=calibration,
        activate_persisted_verified_measurements=False,
    )


def _relation(edge: RelationBelief) -> str:
    if len(edge.relation_probabilities) != 1:
        raise ValueError(f"Phase D edge must have exactly one relation: {edge.edge_id}")
    return next(iter(edge.relation_probabilities))


def _node(overlay: CausalTemporalOverlay, node_id: str):
    return next(node for node in overlay.atomic_events if node.node_id == node_id)


def _variable(edge: RelationBelief, relation: str) -> str:
    return f"relation::{edge.edge_id}::{relation}"


def _labels(result) -> dict[str, str]:
    return {name: label.value for name, label in sorted(result.categorical.items())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overlay", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = run_experiment(args.overlay)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output), "gates": report["gates"]}, indent=2))
    if not report["all_gates_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
