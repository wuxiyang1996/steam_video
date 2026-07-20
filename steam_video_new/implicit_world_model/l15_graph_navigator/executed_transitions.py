"""Auditable real executed-read transition dataset construction."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable

from memory_graph.navigation import GraphReadAction, NavigationActionType

from .artifacts import delta_to_dict
from .belief import FactorizedBeliefBackend
from .contracts import BeliefBackend, BeliefSnapshot, GraphReadExecutor, reasoning_hop_from_action
from .factor_graph import FactorGraphBeliefBackend
from .overlay_io import load_overlay_artifact
from .planner import PersistedGraphReadExecutor, guided_navigation_actions
from .preference_data import require_valid_navigation_case_set
from .realized import derive_realized_belief_delta
from .world_model import evidence_role_for_action


_PACKAGE_DIR = Path(__file__).resolve().parent
_DATASET_SCHEMA = "executed_transition_dataset.schema.json"
_TRUST_STATUSES = {"unreviewed", "ai_provisional", "human_locked"}


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
    grounded_count = 0
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
        actions = guided_navigation_actions(initial, overlay)
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
            grounded = bool(observation_ids) and bool(evidence_refs)
            grounded_count += int(grounded)
            action_name = action.action_type.value
            action_counts[action_name] = action_counts.get(action_name, 0) + 1
            outcome = "observed" if observation_ids else "empty"
            outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1
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
                    "reasoning_hop": reasoning_hop_from_action(action).hop_type.value,
                    "execution": {
                        "skill_id": str(invocation.get("skill_id") or "unknown"),
                        "status": str(invocation.get("status") or "unknown"),
                        "real_observation_ids": list(observation_ids),
                        "evidence_refs": list(evidence_refs),
                        "grounded": grounded,
                        "belief_update_audit": audit,
                    },
                    "target": {
                        "observation_descriptor": {
                            "role": evidence_role_for_action(action).value,
                            "target_ids": list(observation_ids),
                            "node_kind": _node_kind(observations),
                            "predicted_only": False,
                        },
                        "belief_delta": delta_to_dict(delta),
                    },
                    "review_decision": None,
                    "review_rationale": "",
                    "target_source": "executed_read_plus_backend_recompute",
                }
            )

    dataset = {
        "schema_version": "steam-executed-transition-dataset/v0.1",
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
            "action_counts": dict(sorted(action_counts.items())),
            "outcome_counts": dict(sorted(outcome_counts.items())),
            "answerability_counts": dict(sorted(answerability_counts.items())),
            "recovery_counts": dict(sorted(recovery_counts.items())),
            "resolved_role_counts": dict(sorted(resolved_role_counts.items())),
            "hypothesis_disposition_counts": dict(
                sorted(hypothesis_disposition_counts.items())
            ),
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
    for index, record in enumerate(records):
        if record.get("checkpoint_ref") not in checkpoint_ids:
            errors.append(f"records.{index}.checkpoint_ref is unknown")
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
