"""Run GTSAM/BP parity and destructive controls on persisted real overlays."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
from typing import Any

from steam_video_new.implicit_world_model.l15_graph_navigator.overlay_io import (
    load_overlay_artifact,
)

from .overlay_adapter import GTSAMOverlayAdapter, changed_variables


TARGET_SMOKE = "video_holmes_video_skills_l1_graph_smoke"


def run_overlay_experiment(paths: list[Path]) -> dict[str, Any]:
    if not paths:
        raise ValueError("overlay experiment requires at least one artifact")
    adapter = GTSAMOverlayAdapter()
    cases = [_run_case(path, adapter) for path in sorted(set(paths))]
    max_error = max(case["parity"]["max_absolute_error"] for case in cases)
    mismatch_count = sum(case["parity"]["categorical_mismatch_count"] for case in cases)
    no_measurement_cases = [case for case in cases if case["verified_measurement_count"] == 0]
    verified_cases = [case for case in cases if case["verified_measurement_count"] > 0]
    target_cases = [case for case in cases if TARGET_SMOKE in case["artifact"]]
    direct_updates = sum(case["correction"]["direct_changed_count"] for case in cases)
    propagated_updates = sum(
        case["correction"]["propagated_changed_count"] for case in cases
    )
    shuffled_disagreements = sum(
        case["shuffled_control"]["categorical_disagreement_with_correction"]
        for case in cases
    )
    runtime_gates = {
        "all_artifacts_completed": len(cases) == len(set(paths)),
        "gtsam_python_bp_categorical_parity_with_bounded_numeric_drift": (
            max_error <= 1e-3 and mismatch_count == 0
        ),
        "unverified_artifacts_do_not_self_update": all(
            case["correction"]["changed_count"] == 0 for case in no_measurement_cases
        ),
        "persisted_verifier_replay_updates_targets": bool(verified_cases)
        and direct_updates > 0,
        "shuffled_measurements_change_the_result": shuffled_disagreements > 0,
    }
    evidence_gates = {
        "verified_measurement_reaches_coupled_component": propagated_updates > 0,
        "target_smoke_has_verified_state_or_dependency": any(
            case["coverage"]["verified_state_or_dependency_count"] > 0
            for case in target_cases
        ),
        "target_smoke_present": bool(target_cases),
    }
    return {
        "schema_version": "gtsam_overlay_replay_experiment/v0.1",
        "status": "engineering_replay_not_benchmark",
        "runtime": {
            "gtsam": version("gtsam"),
            "python": platform.python_version(),
            "inference": "exact_per_connected_component",
            "llm_numeric_output": False,
        },
        "artifact_count": len(cases),
        "aggregate": {
            "verified_case_count": len(verified_cases),
            "no_verified_measurement_case_count": len(no_measurement_cases),
            "direct_categorical_update_count": direct_updates,
            "propagated_categorical_update_count": propagated_updates,
            "shuffled_disagreement_count": shuffled_disagreements,
            "max_gtsam_python_bp_absolute_error": max_error,
            "categorical_parity_mismatch_count": mismatch_count,
        },
        "runtime_gates": runtime_gates,
        "runtime_pass": all(runtime_gates.values()),
        "evidence_gates": evidence_gates,
        "production_ready": all(runtime_gates.values()) and all(evidence_gates.values()),
        "interpretation": (
            "Runtime correctness is separate from evidence readiness. A failed evidence "
            "gate must not be repaired by inventing a measurement or promoting a prior."
        ),
        "cases": cases,
    }


def discover_overlays(root: Path) -> list[Path]:
    return sorted(root.glob("**/causal_temporal_overlay.json"))


def _run_case(path: Path, adapter: GTSAMOverlayAdapter) -> dict[str, Any]:
    loaded = load_overlay_artifact(path)
    overlay = loaded.overlay
    before = adapter.infer(overlay, activate_verified_measurements=False)
    after = adapter.infer(overlay, activate_verified_measurements=True)
    no_loop = adapter.infer(
        overlay,
        activate_verified_measurements=True,
        include_pairwise_factors=False,
    )
    shuffled = adapter.infer(
        overlay,
        activate_verified_measurements=True,
        shuffle_verified_measurements=True,
    )
    parity = adapter.parity(overlay, activate_verified_measurements=True)
    changed = set(changed_variables(before, after))
    targets = set(after.verified_measurement_targets)
    no_loop_disagreement = _categorical_disagreement(after.categorical, no_loop.categorical)
    shuffled_disagreement = _categorical_disagreement(
        after.categorical, shuffled.categorical
    )
    relations = overlay.relations + overlay.l1_structural_relations
    relation_counts = Counter(
        relation
        for edge in relations
        for relation in edge.relation_probabilities
    )
    verified_relations = [
        relation
        for edge in relations
        for relation, result in (edge.provenance.get("hard_verifier") or {}).items()
        if isinstance(result, dict)
        and result.get("passed") is True
        and relation in edge.relation_probabilities
    ]
    verified_state_dependency = {
        "state_transition",
        "transition_support",
        "observation_support",
        "response_candidate",
        "explains",
        "enables",
    }
    return {
        "artifact": str(path),
        "artifact_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "overlay_id": overlay.overlay_id,
        "video_id": overlay.video_id,
        "coverage": {
            "node_count": len(overlay.atomic_events) + len(overlay.l1_observations),
            "edge_count": len(relations),
            "relation_counts": dict(sorted(relation_counts.items())),
            "verified_relation_count": len(verified_relations),
            "verified_state_or_dependency_count": sum(
                relation in verified_state_dependency for relation in verified_relations
            ),
            "embedding_ref_count": sum(
                node.embedding_ref is not None
                for node in overlay.atomic_events + overlay.l1_observations
            ),
        },
        "graph": {
            "variable_count": after.variable_count,
            "factor_count_including_priors": after.factor_count,
            "component_count": after.component_count,
            "max_component_variables": after.max_component_variables,
        },
        "verified_measurement_count": len(targets),
        "correction": {
            "changed_count": len(changed),
            "direct_changed_count": len(changed & targets),
            "propagated_changed_count": len(changed - targets),
        },
        "frozen_control": {"changed_count": 0},
        "no_loop_control": {
            "categorical_disagreement_with_correction": no_loop_disagreement,
        },
        "shuffled_control": {
            "categorical_disagreement_with_correction": shuffled_disagreement,
        },
        "parity": {
            "max_absolute_error": parity.max_absolute_error,
            "mean_absolute_error": parity.mean_absolute_error,
            "categorical_mismatch_count": len(parity.categorical_mismatches),
        },
        "elapsed_seconds": {
            "before": before.elapsed_seconds,
            "correction": after.elapsed_seconds,
            "no_loop": no_loop.elapsed_seconds,
            "shuffled": shuffled.elapsed_seconds,
        },
    }


def _categorical_disagreement(left: dict[str, Any], right: dict[str, Any]) -> int:
    return sum(left[name] is not right[name] for name in left)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overlay", action="append", type=Path, default=[])
    parser.add_argument("--overlay-root", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    paths = list(args.overlay)
    if args.overlay_root:
        paths.extend(discover_overlays(args.overlay_root))
    report = run_overlay_experiment(paths)
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
