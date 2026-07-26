"""Active-SLAM-style evidence correction pilot using the GTSAM backend."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

from .gtsam_backend import GTSAMDiscreteBeliefGraph, InferenceResult


@dataclass(frozen=True)
class EvidenceQuery:
    query_id: str
    observation_descriptor: str
    expected_belief_delta: str


def build_identity_loop_graph(*, include_loop_closure: bool = True) -> GTSAMDiscreteBeliefGraph:
    graph = GTSAMDiscreteBeliefGraph()
    for variable in ("identity_a_b", "identity_b_c", "conflict_a_c", "decoy_conflict"):
        graph.add_variable(variable)
    graph.add_binary_prior("identity_a_b", 0.93, source="pilot_identity_prior")
    graph.add_binary_prior("identity_b_c", 0.76, source="pilot_identity_prior")
    graph.add_binary_prior("conflict_a_c", 0.08, source="pilot_conflict_prior")
    graph.add_binary_prior("decoy_conflict", 0.08, source="pilot_conflict_prior")
    if include_loop_closure:
        # Assignment order follows GTSAM's listed key order. Only the state
        # identity(A,B)=identity(B,C)=conflict(A,C)=true is incompatible.
        graph.add_factor(
            "identity_loop_closure",
            ("identity_a_b", "identity_b_c", "conflict_a_c"),
            (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.01),
            source="identity_transitivity_and_observed_conflict",
        )
    return graph


def run_correction_pilot(variant: str = "correction") -> dict[str, Any]:
    allowed = {"correction", "frozen", "no_loop_closure", "shuffled_evidence"}
    if variant not in allowed:
        raise ValueError(f"unknown pilot variant {variant!r}; expected {sorted(allowed)}")
    graph = build_identity_loop_graph(include_loop_closure=variant != "no_loop_closure")
    before = graph.infer()
    candidates = (
        EvidenceQuery(
            "inspect_a_c_conflict",
            "inspect whether A and C have incompatible stable attributes or co-occur",
            "may resolve identity-track conflict",
        ),
        EvidenceQuery(
            "repeat_a_b_match",
            "repeat already acquired A-to-B appearance evidence",
            "likely leaves the identity ambiguity unresolved",
        ),
    )
    # This is the planner/world-model boundary: a human or LLM emits only an
    # ordinal label and text rationale. It emits no reward, score, probability,
    # confidence, or utility.
    preference = {
        "left_id": candidates[0].query_id,
        "right_id": candidates[1].query_id,
        "label": "prefer_left",
        "rationale": "left directly tests the unresolved loop-closure conflict",
    }
    selected_query = candidates[0].query_id
    observed = variant != "frozen"
    measurement_target = (
        "decoy_conflict" if variant == "shuffled_evidence" else "conflict_a_c"
    )
    if observed:
        graph.add_binary_measurement(
            "measurement:stable_attribute_conflict",
            measurement_target,
            likelihood_false=0.005,
            likelihood_true=0.995,
            source="pilot_measurement_after_executed_query",
        )
    after = graph.infer()
    public_trace = {
        "candidate_queries": [asdict(candidate) for candidate in candidates],
        "pairwise_preference": preference,
        "selected_query": selected_query,
        "measurement_fixture_received": observed,
        "belief_before": _categorical(after=before),
        "belief_after": _categorical(after=after),
        "replanned_next_step": _next_step(after),
    }
    return {
        "schema_version": "gtsam_active_slam_pilot/v0.1",
        "variant": variant,
        "backend": graph.name,
        "boundary": {
            "llm_output": "categorical transition descriptors and pairwise preference only",
            "numeric_belief_owner": "GTSAM factor graph",
            "query_is_factor": False,
            "measurement_source": "synthetic_controlled_fixture",
            "observed_measurement_becomes_factor": observed,
        },
        "public_categorical_trace": public_trace,
        "solver_audit": {
            "before": _solver_audit(before),
            "after": _solver_audit(after),
            "factor_sources": [factor.source for factor in graph.factors],
        },
    }


def _categorical(*, after: InferenceResult) -> dict[str, str]:
    return {name: label.value for name, label in after.categorical_true.items()}


def _next_step(result: InferenceResult) -> str:
    labels = result.categorical_true
    if labels["conflict_a_c"].value == "accepted":
        return "inspect_identity_b_c_or_split_track"
    if labels["conflict_a_c"].value == "uncertain":
        return "gather_additional_a_c_counterevidence"
    return "continue_with_current_identity_track"


def _solver_audit(result: InferenceResult) -> dict[str, Any]:
    return {
        "map_assignment": result.map_assignment,
        "marginal_true": {name: values[1] for name, values in result.marginals.items()},
        "variable_count": result.variable_count,
        "factor_count": result.factor_count,
        "normalizer": result.normalizer,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant",
        choices=("correction", "frozen", "no_loop_closure", "shuffled_evidence"),
        default="correction",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run_correction_pilot(args.variant)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
