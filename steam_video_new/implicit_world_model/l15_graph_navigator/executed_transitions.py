"""Auditable real executed-read transition dataset construction."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

from memory_graph.navigation import GraphReadAction, NavigationActionType

from .artifacts import delta_to_dict
from .belief import FactorizedBeliefBackend
from .contracts import (
    Answerability,
    BeliefBackend,
    BeliefSnapshot,
    GraphReadExecution,
    GraphReadExecutor,
    UncertaintyLevel,
    reasoning_hop_from_action,
)
from .factor_graph import FactorGraphBeliefBackend
from .overlay_io import load_overlay_artifact
from .planner import PersistedGraphReadExecutor, guided_navigation_actions
from .preference_data import require_valid_navigation_case_set
from .realized import derive_realized_belief_delta
from .world_model import evidence_role_for_action


_PACKAGE_DIR = Path(__file__).resolve().parent
_DATASET_SCHEMA = "executed_transition_dataset.schema.json"
_TRUST_STATUSES = {"unreviewed", "ai_provisional", "human_locked"}
_VERIFIER_APPLICABLE_ACTIONS = {
    NavigationActionType.TEMPORAL_BACK,
    NavigationActionType.TEMPORAL_FORWARD,
    NavigationActionType.TRACK_ENTITY,
    NavigationActionType.INSPECT_STATE_CHANGE,
    NavigationActionType.FOLLOW_DEPENDENCY,
    NavigationActionType.CANDIDATE_CAUSE,
    NavigationActionType.EFFECT,
    NavigationActionType.VERIFY,
}


def build_executed_transition_dataset(
    case_set: dict[str, Any],
    *,
    case_root: Path,
    dataset_id: str,
    belief_backend: str = "factor_graph",
    factor_iterations: int = 8,
    executor_factory: Callable[[], GraphReadExecutor] | None = None,
    include_stop: bool = False,
    allow_ai_provisional: bool = False,
    execution_mode: str = "persisted_replay",
) -> dict[str, Any]:
    """Execute every legal action from a freshly reconstructed checkpoint."""

    require_valid_navigation_case_set(case_set)
    source_status = str(case_set.get("annotation_status") or "")
    if source_status != "human_locked" and not (
        source_status == "ai_provisional" and allow_ai_provisional
    ):
        raise ValueError(
            "transition generation requires a human_locked case set; "
            "pass allow_ai_provisional only for explicitly provisional runs"
        )
    if belief_backend not in {"factor_graph", "factorized"}:
        raise ValueError("belief_backend must be factor_graph or factorized")
    if not dataset_id.strip():
        raise ValueError("dataset_id is required")
    if execution_mode not in {"persisted_replay", "video_skills_runtime"}:
        raise ValueError(
            "execution_mode must be persisted_replay or video_skills_runtime"
        )
    root = case_root.expanduser().resolve()
    make_executor = executor_factory or PersistedGraphReadExecutor
    records: list[dict[str, Any]] = []
    video_ids: set[str] = set()
    action_counts: dict[str, int] = {}
    outcome_counts: dict[str, int] = {}
    answerability_counts: dict[str, int] = {}
    recovery_counts: dict[str, int] = {}
    resolved_role_counts: dict[str, int] = {}
    hypothesis_disposition_counts: dict[str, int] = {}
    verifier_outcome_counts: dict[str, int] = {}
    verifier_source_counts: dict[str, int] = {}
    post_read_verifier_count = 0
    grounded_count = 0
    grounded_empty_result_count = 0
    action_origin_counts: dict[str, int] = {}
    action_legality_counts: dict[str, int] = {}
    checkpoints: list[dict[str, Any]] = []

    def make_backend() -> BeliefBackend:
        if belief_backend == "factor_graph":
            return FactorGraphBeliefBackend(inference_iterations=factor_iterations)
        return FactorizedBeliefBackend()

    for case in sorted(case_set["cases"], key=lambda value: str(value["case_id"])):
        case_id = str(case["case_id"])
        overlay_path = _resolve_path(root, str(case["overlay_path"]))
        loaded = load_overlay_artifact(overlay_path)
        overlay = loaded.overlay
        if overlay.overlay_id != case["overlay_id"]:
            raise ValueError(f"case {case_id} overlay_id does not match its artifact")
        video_ids.add(overlay.video_id)
        initial = make_backend().initialize(
            str(case["question"]),
            overlay,
            seed_evidence=tuple(str(value) for value in case["seed_event_ids"]),
            missing_roles=tuple(str(value) for value in case["missing_roles"]),
            graph_read_budget=int(case["graph_read_budget"]),
        )
        checkpoint_id = _checkpoint_id(case_id, initial)
        native_actions = guided_navigation_actions(initial, overlay)
        reviewed_actions = [
            _reviewed_action(value)
            for value in case.get("acceptable_first_actions") or []
        ]
        actions, provenance_by_key = _merge_actions(
            native_actions,
            reviewed_actions,
            overlay,
            source_status=source_status,
        )
        checkpoints.append(
            {
                "case_id": case_id,
                "checkpoint_id": checkpoint_id,
                "checkpoint": _categorical_belief(initial),
            }
        )
        if not include_stop:
            actions = [
                action
                for action in actions
                if action.action_type is not NavigationActionType.STOP
            ]
        actions = sorted(actions, key=_action_key)
        for action in actions:
            # A fresh backend and executor prevent cross-sibling belief leakage.
            backend = make_backend()
            before = backend.initialize(
                str(case["question"]),
                overlay,
                seed_evidence=tuple(str(value) for value in case["seed_event_ids"]),
                missing_roles=tuple(str(value) for value in case["missing_roles"]),
                graph_read_budget=int(case["graph_read_budget"]),
            )
            executor = make_executor()
            if action.action_type is NavigationActionType.STOP:
                observations = ()
                invocation: dict[str, object] = {
                    "skill_id": "executed_transition_stop",
                    "status": "executed",
                    "evidence_refs": [],
                }
                after = before
                audit = None
            else:
                provenance = provenance_by_key[_action_key(action)]
                execute_review_anchored = getattr(
                    executor, "execute_review_anchored", None
                )
                if (
                    provenance["origin"] == "review_anchored"
                    and provenance["legality"] != "not_executable"
                    and callable(execute_review_anchored)
                ):
                    execution = execute_review_anchored(before, action, overlay)
                elif provenance["legality"] == "not_executable":
                    execution = GraphReadExecution(
                        (),
                        {
                            "skill_id": "review_anchored_endpoint_read",
                            "status": "failed",
                            "failure_code": "reviewed_action_endpoint_not_executable",
                            "evidence_refs": [],
                            "grounded_empty_result": False,
                            "search_completed": False,
                        },
                    )
                else:
                    execution = executor.execute(before, action, overlay)
                observations = execution.observations
                update_from_execution = getattr(backend, "update_from_execution", None)
                update = (
                    update_from_execution(before, action, execution, overlay)
                    if callable(update_from_execution)
                    else backend.update(before, action, list(observations), overlay)
                )
                invocation = execution.skill_invocation
                after = update.belief
                audit = update.audit_record
                after, semantic_audit = _apply_executed_operation_semantics(
                    before=before,
                    after=after,
                    case=case,
                    action=action,
                    invocation=invocation,
                    observations=observations,
                    provenance=provenance,
                )
                if semantic_audit:
                    audit = {
                        "backend_update": audit,
                        "executed_operation_correction": semantic_audit,
                    }
            delta = derive_realized_belief_delta(before, after)
            answerability = delta.answerability_after.value
            answerability_counts[answerability] = answerability_counts.get(answerability, 0) + 1
            recovery = delta.recovery_status.value
            recovery_counts[recovery] = recovery_counts.get(recovery, 0) + 1
            for role in delta.resolved_roles:
                resolved_role_counts[role] = resolved_role_counts.get(role, 0) + 1
            for update in delta.hypothesis_updates:
                disposition = update.disposition.value
                hypothesis_disposition_counts[disposition] = (
                    hypothesis_disposition_counts.get(disposition, 0) + 1
                )
            observation_ids = tuple(node.node_id for node in observations)
            known_ids = {
                node.node_id
                for node in overlay.atomic_events + overlay.l1_observations
            }
            if not set(observation_ids) <= known_ids:
                raise ValueError(f"case {case_id} execution returned unknown observation")
            evidence_refs = tuple(
                dict.fromkeys(
                    str(value)
                    for value in (invocation.get("evidence_refs") or [])
                )
            )
            verifier = _verifier_measurement(invocation, action)
            verifier_outcome = str(verifier["categorical_outcome"])
            verifier_source = str(verifier["source"])
            verifier_outcome_counts[verifier_outcome] = (
                verifier_outcome_counts.get(verifier_outcome, 0) + 1
            )
            verifier_source_counts[verifier_source] = (
                verifier_source_counts.get(verifier_source, 0) + 1
            )
            post_read_verifier_count += int(verifier["post_read"] is True)
            grounded_empty_result = invocation.get("grounded_empty_result") is True
            grounded = (
                bool(observation_ids) and bool(evidence_refs)
            ) or grounded_empty_result
            grounded_count += int(grounded)
            grounded_empty_result_count += int(grounded_empty_result)
            action_name = action.action_type.value
            action_counts[action_name] = action_counts.get(action_name, 0) + 1
            outcome = (
                "observed"
                if observation_ids
                else "empty_search_observed"
                if grounded_empty_result
                else "empty"
            )
            outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1
            provenance = provenance_by_key[_action_key(action)]
            origin = str(provenance["origin"])
            legality = str(provenance["legality"])
            action_origin_counts[origin] = action_origin_counts.get(origin, 0) + 1
            action_legality_counts[legality] = action_legality_counts.get(legality, 0) + 1
            records.append(
                {
                    "record_id": _record_id(case_id, action),
                    "case_id": case_id,
                    "video_id": overlay.video_id,
                    "overlay_id": overlay.overlay_id,
                    "overlay_sha256": _sha256(overlay_path),
                    "question": str(case["question"]),
                    "checkpoint_ref": checkpoint_id,
                    "local_context": _local_context(action, overlay),
                    "action": _action_payload(action),
                    "action_provenance": provenance,
                    "reasoning_hop": reasoning_hop_from_action(action).hop_type.value,
                    "execution": {
                        "skill_id": str(invocation.get("skill_id") or "unknown"),
                        "status": str(invocation.get("status") or "unknown"),
                        "real_observation_ids": list(observation_ids),
                        "evidence_refs": list(evidence_refs),
                        "grounded": grounded,
                        "observation_outcome": outcome,
                        "grounded_empty_result": grounded_empty_result,
                        "belief_update_audit": audit,
                        "verifier_measurement": verifier,
                    },
                    "target": {
                        "observation_descriptor": {
                            "role": evidence_role_for_action(action).value,
                            "target_ids": list(observation_ids),
                            "node_kind": (
                                "empty_counterevidence_search"
                                if grounded_empty_result
                                else _node_kind(observations)
                            ),
                            "predicted_only": False,
                        },
                        "belief_delta": delta_to_dict(delta),
                    },
                    "review_decision": None,
                    "review_rationale": "",
                    "target_source": "executed_read_plus_backend_recompute_v2",
                }
            )

    dataset = {
        "schema_version": "steam-executed-transition-dataset/v0.2",
        "dataset_id": dataset_id,
        "annotation_status": "unreviewed",
        "annotator": None,
        "locked_sha256": None,
        "source_case_set": {
            "case_set_id": case_set["case_set_id"],
            "annotation_status": source_status,
            "locked_sha256": case_set.get("locked_sha256"),
        },
        "belief_backend": belief_backend,
        "execution_mode": execution_mode,
        "formal_eligible": False,
        "numeric_model_output": False,
        "summary": {
            "case_count": len(case_set["cases"]),
            "video_count": len(video_ids),
            "record_count": len(records),
            "grounded_record_count": grounded_count,
            "grounded_empty_result_count": grounded_empty_result_count,
            "action_origin_counts": dict(sorted(action_origin_counts.items())),
            "action_legality_counts": dict(sorted(action_legality_counts.items())),
            "action_counts": dict(sorted(action_counts.items())),
            "outcome_counts": dict(sorted(outcome_counts.items())),
            "answerability_counts": dict(sorted(answerability_counts.items())),
            "recovery_counts": dict(sorted(recovery_counts.items())),
            "resolved_role_counts": dict(sorted(resolved_role_counts.items())),
            "hypothesis_disposition_counts": dict(
                sorted(hypothesis_disposition_counts.items())
            ),
            "verifier_outcome_counts": dict(sorted(verifier_outcome_counts.items())),
            "verifier_source_counts": dict(sorted(verifier_source_counts.items())),
            "post_read_verifier_count": post_read_verifier_count,
        },
        "checkpoints": checkpoints,
        "records": records,
    }
    require_valid_executed_transition_dataset(dataset)
    return dataset


def validate_executed_transition_dataset(payload: dict[str, Any]) -> list[str]:
    try:
        import jsonschema
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("jsonschema is required for transition datasets") from exc
    schema = json.loads((_PACKAGE_DIR / _DATASET_SCHEMA).read_text(encoding="utf-8"))
    errors = [
        f"{'.'.join(str(part) for part in error.absolute_path) or '<root>'}: {error.message}"
        for error in sorted(
            jsonschema.Draft202012Validator(schema).iter_errors(payload),
            key=lambda error: list(error.path),
        )
    ]
    records = payload.get("records") or []
    checkpoints = payload.get("checkpoints") or []
    checkpoint_ids = {
        str(value.get("checkpoint_id") or "")
        for value in checkpoints
    }
    ids = [str(record.get("record_id") or "") for record in records]
    if len(ids) != len(set(ids)):
        errors.append("record_id values must be unique")
    is_v2 = payload.get("schema_version") == "steam-executed-transition-dataset/v0.2"
    if is_v2:
        for field in (
            "grounded_empty_result_count",
            "action_origin_counts",
            "action_legality_counts",
        ):
            if field not in (payload.get("summary") or {}):
                errors.append(f"summary.{field} is required for v0.2")
    for index, record in enumerate(records):
        if record.get("checkpoint_ref") not in checkpoint_ids:
            errors.append(f"records.{index}.checkpoint_ref is unknown")
        if is_v2:
            if "action_provenance" not in record:
                errors.append(f"records.{index}.action_provenance is required for v0.2")
            execution = record.get("execution") or {}
            for field in ("observation_outcome", "grounded_empty_result"):
                if field not in execution:
                    errors.append(f"records.{index}.execution.{field} is required for v0.2")
        target = record.get("target") or {}
        observation = target.get("observation_descriptor") or {}
        delta = target.get("belief_delta") or {}
        if observation.get("predicted_only") is not False:
            errors.append(f"records.{index} observation is not real")
        if delta.get("predicted_only") is not False:
            errors.append(f"records.{index} belief delta is not real")
        numeric_paths = _numeric_paths(target, path=f"records.{index}.target")
        if numeric_paths:
            errors.append(
                f"records.{index} target contains numeric values: {numeric_paths[:3]}"
            )
    status = str(payload.get("annotation_status") or "")
    if status in {"ai_provisional", "human_locked"}:
        for index, record in enumerate(records):
            if record.get("review_decision") not in {"accept", "reject"}:
                errors.append(f"records.{index}.review_decision is incomplete")
            if not str(record.get("review_rationale") or "").strip():
                errors.append(f"records.{index}.review_rationale is empty")
    checksum = payload.get("locked_sha256")
    if status == "human_locked" and not checksum:
        errors.append("human_locked transition datasets require locked_sha256")
    if checksum and checksum != _content_checksum(payload):
        errors.append("transition dataset locked_sha256 does not match content")
    if payload.get("formal_eligible") is not (status == "human_locked"):
        errors.append("formal_eligible must be true exactly for human_locked datasets")
    return errors


def require_valid_executed_transition_dataset(payload: dict[str, Any]) -> None:
    errors = validate_executed_transition_dataset(payload)
    if errors:
        raise ValueError("invalid executed transition dataset: " + "; ".join(errors[:8]))


def lock_executed_transition_dataset(
    payload: dict[str, Any],
    *,
    annotation_status: str,
    annotator: str,
) -> dict[str, Any]:
    if annotation_status not in {"ai_provisional", "human_locked"}:
        raise ValueError("annotation_status must be ai_provisional or human_locked")
    locked = json.loads(json.dumps(payload))
    locked["annotation_status"] = annotation_status
    locked["annotator"] = annotator
    locked["formal_eligible"] = annotation_status == "human_locked"
    locked["locked_sha256"] = None
    errors = validate_executed_transition_dataset(locked)
    errors = [error for error in errors if "locked_sha256" not in error]
    if errors:
        raise ValueError("cannot lock executed transition dataset: " + "; ".join(errors[:8]))
    locked["locked_sha256"] = _content_checksum(locked)
    require_valid_executed_transition_dataset(locked)
    return locked


def export_executed_transition_training_records(
    payload: dict[str, Any],
    *,
    allow_ai_provisional: bool = False,
) -> list[dict[str, Any]]:
    require_valid_executed_transition_dataset(payload)
    status = str(payload["annotation_status"])
    if status != "human_locked" and not (
        status == "ai_provisional" and allow_ai_provisional
    ):
        raise ValueError(
            "transition export requires human_locked review; pass "
            "allow_ai_provisional only for provisional experiments"
        )
    checkpoints = {
        str(row["checkpoint_id"]): row["checkpoint"]
        for row in payload["checkpoints"]
    }
    return [
        {
            "task": "observation_belief_transition",
            "record_id": record["record_id"],
            "case_id": record["case_id"],
            "video_id": record["video_id"],
            "question": record["question"],
            "checkpoint": checkpoints[record["checkpoint_ref"]],
            "local_context": record["local_context"],
            "action": record["action"],
            "action_provenance": record.get("action_provenance"),
            "execution_provenance": {
                "skill_id": record["execution"]["skill_id"],
                "grounded": record["execution"]["grounded"],
                "observation_outcome": record["execution"].get(
                    "observation_outcome", "legacy_unspecified"
                ),
                "grounded_empty_result": record["execution"].get(
                    "grounded_empty_result", False
                ),
                "verifier_measurement": record["execution"][
                    "verifier_measurement"
                ],
            },
            "target": record["target"],
            "label_source": status,
        }
        for record in payload["records"]
        if record["review_decision"] == "accept"
    ]


def _categorical_belief(belief: BeliefSnapshot) -> dict[str, Any]:
    return {
        "belief_id": belief.belief_id,
        "acquired_evidence": sorted(belief.acquired_evidence),
        "frontier": sorted(belief.frontier),
        "missing_roles": sorted(belief.missing_roles),
        "contradictions": sorted(belief.contradictions),
        "hypotheses": [
            {"edge_id": state.edge_id, "grounding": state.grounding.value}
            for state in sorted(belief.relation_states, key=lambda value: value.edge_id)
        ],
        "blocked_edge_ids": sorted(belief.blocked_edge_ids),
        "uncertainty": belief.uncertainty.value,
        "answerability": belief.answerability.value,
        "read_budget_state": (
            "available" if belief.remaining_graph_reads else "exhausted"
        ),
    }


def _local_context(action: GraphReadAction, overlay) -> dict[str, Any]:
    ids = set(action.target_ids)
    if action.source_id:
        ids.add(action.source_id)
    nodes = [
        node
        for node in overlay.atomic_events + overlay.l1_observations
        if node.node_id in ids
    ]
    edges = [
        edge
        for edge in overlay.relations + overlay.l1_structural_relations
        if edge.src in ids and edge.dst in ids
    ]
    return {
        "nodes": [
            {
                "node_id": node.node_id,
                "node_kind": node.node_type,
                "description": node.text,
                "evidence_refs": list(node.source_segments),
                "embedding_ref": (
                    {
                        "path": node.embedding_ref.path,
                        "model": node.embedding_ref.model,
                        "dimension": node.embedding_ref.dimension,
                        "dtype": node.embedding_ref.dtype,
                        "normalized": node.embedding_ref.normalized,
                        "row_index": node.embedding_ref.row_index,
                        "checksum": node.embedding_ref.checksum,
                    }
                    if node.embedding_ref is not None
                    else None
                ),
            }
            for node in sorted(nodes, key=lambda value: value.node_id)
        ],
        "relations": [
            {
                "edge_id": edge.edge_id,
                "src": edge.src,
                "dst": edge.dst,
                "relation_types": sorted(edge.relation_probabilities),
            }
            for edge in sorted(edges, key=lambda value: value.edge_id)
        ],
    }


def _action_payload(action: GraphReadAction) -> dict[str, Any]:
    return {
        "action_type": action.action_type.value,
        "source_id": action.source_id,
        "target_ids": list(action.target_ids),
        "relation": action.relation,
    }


def _reviewed_action(payload: dict[str, Any]) -> GraphReadAction:
    return GraphReadAction(
        action_type=NavigationActionType(str(payload["action_type"])),
        source_id=(
            str(payload["source_id"])
            if payload.get("source_id") is not None
            else None
        ),
        target_ids=tuple(str(value) for value in payload.get("target_ids") or []),
        relation=(
            str(payload["relation"])
            if payload.get("relation") is not None
            else None
        ),
        rationale="independently reviewed acceptable first action",
    )


def _merge_actions(
    native_actions: list[GraphReadAction],
    reviewed_actions: list[GraphReadAction],
    overlay,
    *,
    source_status: str,
) -> tuple[list[GraphReadAction], dict[str, dict[str, Any]]]:
    native = {_action_key(action): action for action in native_actions}
    reviewed = {_action_key(action): action for action in reviewed_actions}
    merged = {**native, **reviewed}
    provenance: dict[str, dict[str, Any]] = {}
    for key, action in merged.items():
        is_native = key in native
        is_reviewed = key in reviewed
        if is_native and is_reviewed:
            origin = "review_anchored_and_native"
        elif is_reviewed:
            origin = "review_anchored"
        elif action.action_type is NavigationActionType.STOP:
            origin = "control"
        else:
            origin = "native_candidate"
        legality, edge_admitted = _action_legality(
            action,
            overlay,
            native_legal=is_native,
            reviewed=is_reviewed,
        )
        provenance[key] = {
            "origin": origin,
            "legality": legality,
            "native_candidate": is_native,
            "reviewed_accepted": is_reviewed,
            "review_status": source_status if is_reviewed else None,
            "edge_admitted": edge_admitted,
            "graph_mutated": False,
        }
    return list(merged.values()), provenance


def _action_legality(
    action: GraphReadAction,
    overlay,
    *,
    native_legal: bool,
    reviewed: bool,
) -> tuple[str, bool]:
    edge_admitted = _matching_admitted_edge(action, overlay)
    if native_legal:
        return "native_legal", edge_admitted
    if action.action_type is NavigationActionType.STOP:
        return "not_applicable", False
    known = {
        node.node_id for node in overlay.atomic_events + overlay.l1_observations
    }
    source_known = action.source_id is None or action.source_id in known
    targets_known = all(target in known for target in action.target_ids)
    targetless_allowed = action.action_type in {
        NavigationActionType.SEMANTIC,
        NavigationActionType.SEARCH_COUNTEREVIDENCE,
    }
    if not source_known or (not action.target_ids and not targetless_allowed) or not targets_known:
        return "not_executable", edge_admitted
    if reviewed and action.action_type is NavigationActionType.VERIFY and not edge_admitted:
        return "offline_diagnostic", False
    return "review_restored", edge_admitted


def _matching_admitted_edge(action: GraphReadAction, overlay) -> bool:
    if not action.source_id or not action.target_ids or not action.relation:
        return False
    for edge in overlay.relations + overlay.l1_structural_relations:
        if action.source_id not in {edge.src, edge.dst}:
            continue
        if not any(target in {edge.src, edge.dst} for target in action.target_ids):
            continue
        if action.relation in edge.relation_probabilities:
            return True
        if action.relation == "event_projection" and edge.provenance == "l1_grounding":
            return True
    return False


def _apply_executed_operation_semantics(
    *,
    before: BeliefSnapshot,
    after: BeliefSnapshot,
    case: dict[str, Any],
    action: GraphReadAction,
    invocation: dict[str, object],
    observations: tuple[Any, ...],
    provenance: dict[str, Any],
) -> tuple[BeliefSnapshot, dict[str, object] | None]:
    """Apply categorical semantics of a completed reviewed operation.

    This correction never admits an edge and never consumes the reviewer's
    outcome label.  It only marks the requested evidence role as inspected
    when the real execution contract itself establishes completion.
    """

    if provenance.get("reviewed_accepted") is not True:
        return after, None
    requested_roles = tuple(str(value) for value in case.get("missing_roles") or [])
    verifier = invocation.get("verifier_result") or {}
    verifier_outcome = (
        verifier.get("categorical_outcome") if isinstance(verifier, dict) else None
    )
    completed_role: str | None = None
    reason = ""
    if (
        "counterevidence" in requested_roles
        and action.action_type is NavigationActionType.SEARCH_COUNTEREVIDENCE
        and invocation.get("search_completed") is True
    ):
        completed_role = "counterevidence"
        reason = "real counterevidence search completed, including a grounded empty result"
    elif (
        "state_transition" in requested_roles
        and verifier_outcome == "supports"
        and action.action_type
        in {NavigationActionType.INSPECT_STATE_CHANGE, NavigationActionType.VERIFY}
    ):
        completed_role = "state_transition"
        reason = "post-read categorical verifier established a strict grounded state delta"
    elif (
        "bridge" in requested_roles
        and observations
        and action.action_type
        in {
            NavigationActionType.FIND_BRIDGE,
            NavigationActionType.TEMPORAL_BACK,
            NavigationActionType.TEMPORAL_FORWARD,
            NavigationActionType.TRACK_ENTITY,
            NavigationActionType.FOLLOW_DEPENDENCY,
        }
    ):
        completed_role = "bridge"
        reason = "reviewed first-hop endpoint was actually read and opened the bridge frontier"
    if completed_role is None or completed_role not in after.missing_roles:
        return after, None
    missing = tuple(role for role in after.missing_roles if role != completed_role)
    uncertainty = UncertaintyLevel.LOW if not missing else UncertaintyLevel.MEDIUM
    answerability = (
        Answerability.READY
        if after.acquired_evidence and not missing and not after.contradictions
        else Answerability.NOT_READY
    )
    corrected = replace(
        after,
        missing_roles=missing,
        uncertainty=uncertainty,
        answerability=answerability,
    )
    return corrected, {
        "role": completed_role,
        "reason": reason,
        "source": "executed_operation_categorical_semantics",
        "graph_edge_inserted": False,
        "review_outcome_used_as_measurement": False,
        "numeric_output_exposed": False,
    }


def _action_key(action: GraphReadAction) -> str:
    return json.dumps(_action_payload(action), sort_keys=True, separators=(",", ":"))


def _record_id(case_id: str, action: GraphReadAction) -> str:
    digest = hashlib.sha256(
        f"{case_id}\0{_action_key(action)}".encode("utf-8")
    ).hexdigest()[:24]
    return f"transition:{digest}"


def _checkpoint_id(case_id: str, belief: BeliefSnapshot) -> str:
    digest = hashlib.sha256(
        f"{case_id}\0{belief.belief_id}".encode("utf-8")
    ).hexdigest()[:24]
    return f"checkpoint:{digest}"


def _node_kind(observations) -> str:
    kinds = sorted({node.node_type for node in observations})
    if not kinds:
        return "none"
    if len(kinds) == 1:
        return kinds[0]
    return "mixed"


def _resolve_path(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _content_checksum(payload: dict[str, Any]) -> str:
    clone = json.loads(json.dumps(payload))
    clone["locked_sha256"] = None
    encoded = json.dumps(
        clone,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _numeric_paths(value: Any, *, path: str) -> list[str]:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return []
    if isinstance(value, (int, float)):
        return [path]
    if isinstance(value, dict):
        return [
            child_path
            for key, child in value.items()
            for child_path in _numeric_paths(child, path=f"{path}.{key}")
        ]
    if isinstance(value, list):
        return [
            child_path
            for index, child in enumerate(value)
            for child_path in _numeric_paths(child, path=f"{path}[{index}]")
        ]
    return [path]


def _verifier_measurement(
    invocation: dict[str, object],
    action: GraphReadAction,
) -> dict[str, Any]:
    value = invocation.get("verifier_result") or {}
    if not isinstance(value, dict):
        value = {}
    outcome = value.get("categorical_outcome")
    if outcome not in {"supports", "rejects", "inconclusive", "not_applicable"}:
        outcome = "inconclusive"
    applicability = value.get("applicability")
    if applicability not in {"applicable", "not_applicable"}:
        applicability = (
            "applicable"
            if action.action_type in _VERIFIER_APPLICABLE_ACTIONS
            else "not_applicable"
        )
    if applicability == "not_applicable":
        outcome = "not_applicable"
    return {
        "applicability": applicability,
        "categorical_outcome": outcome,
        "source": str(value.get("source") or "missing_verifier_contract"),
        "post_read": value.get("post_read") is True,
        "evidence_refs": list(
            dict.fromkeys(str(item) for item in (value.get("evidence_refs") or []))
        ),
        "reasons": list(
            dict.fromkeys(str(item) for item in (value.get("reasons") or []))
        ),
        "numeric_output_exposed": False,
    }
