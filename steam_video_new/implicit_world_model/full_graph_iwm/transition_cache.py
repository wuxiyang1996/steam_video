"""Persistent, leakage-auditable cache for action-conditioned IWM predictions."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from enum import Enum
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Sequence

from .contracts import (
    AnswerabilityState,
    BatchedCategoricalWorldModel,
    CategoricalBeliefDelta,
    ContradictionChange,
    EvidenceOutcome,
    FrontierChange,
    IWMRequest,
    ImaginedTransition,
    PredictedObservation,
    ProgressChange,
)
from .model_input import graph_input_to_categorical_payload


CACHE_SCHEMA = "steam-action-conditioned-transition-cache/v0.2"
ROLE_CACHE_SCHEMA = "steam-question-role-cache/v0.1"
RESPONSE_CACHE_SCHEMA = "steam-categorical-response-cache/v0.1"


class PersistentCategoricalResponseCacheClient:
    """Record/replay strict categorical model calls without storing prompts."""

    def __init__(self, delegate: Any, path: Path, *, mode: str = "record") -> None:
        if mode not in {"record", "replay"}:
            raise ValueError("response cache mode must be record or replay")
        self.delegate = delegate
        self.path = path.expanduser().resolve()
        self.mode = mode
        self.model = str(delegate.model)
        self._entries = self._load()
        self.response_audits: list[dict[str, Any]] = []
        self.last_response_audit: dict[str, Any] = {}

    def complete_json(self, *, task: str, payload: dict[str, Any]) -> dict[str, Any]:
        request_payload = {"model": self.model, "task": task, "payload": payload}
        request_sha = _checksum(request_payload)
        key = request_sha
        cached = self._entries.get(key)
        if cached is None:
            if self.mode == "replay":
                raise RuntimeError(f"frozen categorical response cache miss: {key[:12]}")
            response = self.delegate.complete_json(task=task, payload=payload)
            if _contains_number(response):
                raise ValueError("categorical response cache refuses numeric model output")
            cached = {"request_sha256": request_sha, "response": response}
            self._entries[key] = cached
            self._persist()
            delegate_audit = dict(getattr(self.delegate, "last_response_audit", {}))
            audit = {"cache_hit": False, **delegate_audit}
        else:
            response = cached.get("response")
            if not isinstance(response, dict) or _contains_number(response):
                raise ValueError("categorical response cache entry is invalid")
            audit = {"cache_hit": True}
        self.last_response_audit = audit
        self.response_audits.append(dict(audit))
        return json.loads(json.dumps(response))

    def _load(self) -> dict[str, dict[str, Any]]:
        if not self.path.exists():
            if self.mode == "replay":
                raise FileNotFoundError(f"categorical response cache is missing: {self.path}")
            return {}
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != RESPONSE_CACHE_SCHEMA:
            raise ValueError("categorical response cache schema mismatch")
        if payload.get("model") != self.model:
            raise ValueError("categorical response cache model mismatch")
        entries = payload.get("entries")
        if not isinstance(entries, dict):
            raise ValueError("categorical response cache entries must be an object")
        return entries

    def _persist(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        value = {
            "schema_version": RESPONSE_CACHE_SCHEMA,
            "model": self.model,
            "request_payload_stored": False,
            "hidden_clue_or_answer_used": False,
            "numeric_reward_present": False,
            "training_performed": False,
            "entry_count": len(self._entries),
            "entries": {key: self._entries[key] for key in sorted(self._entries)},
        }
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, self.path)


class PersistentQuestionRoleCache:
    """Record/replay the categorical question decomposition across retries."""

    def __init__(self, delegate: Any, path: Path, *, mode: str = "record") -> None:
        if mode not in {"record", "replay"}:
            raise ValueError("question role cache mode must be record or replay")
        self.delegate = delegate
        self.path = path.expanduser().resolve()
        self.mode = mode
        self.model_name = str(delegate.model_name)
        self._entries = self._load()

    def initialize(self, question: str) -> tuple[str, ...]:
        key = hashlib.sha256(
            f"{self.model_name}\x1f{question}".encode("utf-8")
        ).hexdigest()
        if key in self._entries:
            return _string_tuple(self._entries[key]["missing_roles"], "missing_roles")
        if self.mode == "replay":
            raise RuntimeError("frozen question role cache miss")
        roles = tuple(self.delegate.initialize(question))
        self._entries[key] = {
            "question_sha256": hashlib.sha256(question.encode("utf-8")).hexdigest(),
            "missing_roles": list(roles),
        }
        self._persist()
        return roles

    def _load(self) -> dict[str, dict[str, Any]]:
        if not self.path.exists():
            if self.mode == "replay":
                raise FileNotFoundError(f"question role cache is missing: {self.path}")
            return {}
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != ROLE_CACHE_SCHEMA:
            raise ValueError("question role cache schema mismatch")
        if payload.get("model") != self.model_name:
            raise ValueError("question role cache model mismatch")
        entries = payload.get("entries")
        if not isinstance(entries, dict):
            raise ValueError("question role cache entries must be an object")
        return entries

    def _persist(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": ROLE_CACHE_SCHEMA,
            "model": self.model_name,
            "question_text_stored": False,
            "hidden_clue_or_answer_used": False,
            "training_performed": False,
            "entry_count": len(self._entries),
            "entries": {key: self._entries[key] for key in sorted(self._entries)},
        }
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, self.path)


class PersistentTransitionCacheWorldModel(BatchedCategoricalWorldModel):
    """Record or replay categorical transitions using the complete safe request.

    The key covers the current visible graph view, real belief, legal action,
    parent actions, and imagined prefix. It never uses clue intervals, answers,
    or evaluator labels. Replay mode fails closed on a miss.
    """

    def __init__(
        self,
        delegate: BatchedCategoricalWorldModel,
        path: Path,
        *,
        mode: str = "record",
    ) -> None:
        if mode not in {"record", "replay"}:
            raise ValueError("transition cache mode must be record or replay")
        self.delegate = delegate
        self.path = path.expanduser().resolve()
        self.mode = mode
        self.delegate_model_name = str(delegate.model_name)
        self.model_name = f"transition-cache:{mode}:{self.delegate_model_name}"
        self.persist_batch_size = max(1, int(getattr(delegate, "batch_size", 48)))
        self._entries = self._load()
        self.cache_audits: list[dict[str, int | str]] = []

    def predict_batch(
        self,
        requests: Sequence[IWMRequest],
    ) -> Sequence[ImaginedTransition]:
        keys = tuple(_request_key(request, self.delegate_model_name) for request in requests)
        missing_indices = [
            index for index, key in enumerate(keys) if key not in self._entries
        ]
        if missing_indices and self.mode == "replay":
            missing_preview = ", ".join(keys[index][:12] for index in missing_indices[:4])
            raise RuntimeError(
                "frozen transition cache miss; gather before evaluation: "
                f"{missing_preview}"
            )
        if missing_indices:
            for start in range(0, len(missing_indices), self.persist_batch_size):
                batch_indices = missing_indices[start : start + self.persist_batch_size]
                batch_requests = tuple(requests[index] for index in batch_indices)
                predicted = tuple(self.delegate.predict_batch(batch_requests))
                if len(predicted) != len(batch_requests):
                    raise ValueError("transition-cache delegate coverage mismatch")
                for index, transition in zip(batch_indices, predicted):
                    request = requests[index]
                    if transition.action != request.action:
                        raise ValueError("transition-cache delegate changed the legal action")
                    self._entries[keys[index]] = {
                        "request_audit": _request_audit(request),
                        "transition": _transition_to_dict(transition),
                    }
                # Long horizon-two gathers may take hundreds of service calls.
                # Commit every transport-sized batch so interruption is resumable.
                self._persist()
        result = tuple(
            _transition_from_dict(self._entries[key]["transition"], request)
            for key, request in zip(keys, requests)
        )
        self.cache_audits.append(
            {
                "mode": self.mode,
                "request_count": len(requests),
                "hit_count": len(requests) - len(missing_indices),
                "miss_count": len(missing_indices),
                "entry_count_after": len(self._entries),
            }
        )
        return result

    @property
    def entry_count(self) -> int:
        return len(self._entries)

    def _load(self) -> dict[str, dict[str, Any]]:
        if not self.path.exists():
            if self.mode == "replay":
                raise FileNotFoundError(f"transition cache is missing: {self.path}")
            return {}
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != CACHE_SCHEMA:
            raise ValueError("transition cache schema mismatch")
        if payload.get("model") != self.delegate_model_name:
            raise ValueError("transition cache model mismatch")
        entries = payload.get("entries")
        if not isinstance(entries, dict):
            raise ValueError("transition cache entries must be an object")
        if int(payload.get("entry_count", -1)) != len(entries):
            raise ValueError("transition cache entry count mismatch")
        return entries

    def _persist(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": CACHE_SCHEMA,
            "model": self.delegate_model_name,
            "cache_key_contract": (
                "safe graph input + real belief + legal action + imagined prefix"
            ),
            "hidden_clue_or_answer_used": False,
            "numeric_reward_present": False,
            "training_performed": False,
            "entry_count": len(self._entries),
            "entries": {key: self._entries[key] for key in sorted(self._entries)},
        }
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, self.path)


def _request_key(request: IWMRequest, model_name: str) -> str:
    payload = {
        "schema_version": CACHE_SCHEMA,
        "model": model_name,
        "request": _request_payload(request),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _request_audit(request: IWMRequest) -> dict[str, Any]:
    payload = _request_payload(request)
    return {
        "belief_id": request.belief.belief_id,
        "action_id": request.action.action_id,
        "action_kind": request.action.kind.value,
        "target_id": request.action.target_id,
        "parent_action_ids": list(request.parent_action_ids),
        "imagined_prefix_length": len(request.imagined_history),
        "question_sha256": hashlib.sha256(
            request.belief.question.encode("utf-8")
        ).hexdigest(),
        "request_sha256": hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "hidden_evaluator_fields_present": False,
    }


def _request_payload(request: IWMRequest) -> dict[str, Any]:
    return {
        # belief_id contains the experiment arm/run name and is not part of the
        # semantic state. Excluding it is required for matched-arm reuse.
        "belief": {
            "question": request.belief.question,
            "current_node_id": request.belief.current_node_id,
            "acquired_evidence": list(request.belief.acquired_evidence),
            "imagined_evidence": list(request.belief.imagined_evidence),
            "cursor_history": list(request.belief.cursor_history),
            "accepted_relations": list(request.belief.accepted_relations),
            "rejected_relations": list(request.belief.rejected_relations),
            "unresolved_relations": list(request.belief.unresolved_relations),
            "missing_roles": list(request.belief.missing_roles),
            "contradictions": list(request.belief.contradictions),
            "answerability": request.belief.answerability.value,
            "remaining_reads": request.belief.remaining_reads,
            "step": request.belief.step,
        },
        "graph_input": graph_input_to_categorical_payload(request.graph_input),
        "action": _jsonable(request.action),
        "parent_action_ids": list(request.parent_action_ids),
        "imagined_history": [_transition_to_dict(row) for row in request.imagined_history],
    }


def _transition_to_dict(transition: ImaginedTransition) -> dict[str, Any]:
    return {
        "action_id": transition.action.action_id,
        "observation": {
            "target_id": transition.observation.target_id,
            "outcome": transition.observation.outcome.value,
            "descriptor": list(transition.observation.descriptor),
            "predicted_only": True,
        },
        "belief_delta": {
            "progress": transition.belief_delta.progress.value,
            "answerability_after": transition.belief_delta.answerability_after.value,
            "frontier_change": transition.belief_delta.frontier_change.value,
            "contradiction_change": (
                transition.belief_delta.contradiction_change.value
            ),
            "resolved_roles": list(transition.belief_delta.resolved_roles),
            "opened_roles": list(transition.belief_delta.opened_roles),
            "relation_updates": list(transition.belief_delta.relation_updates),
            "predicted_only": True,
        },
    }


def _transition_from_dict(
    payload: dict[str, Any],
    request: IWMRequest,
) -> ImaginedTransition:
    if payload.get("action_id") != request.action.action_id:
        raise ValueError("cached transition action mismatch")
    observation = payload.get("observation") or {}
    delta = payload.get("belief_delta") or {}
    descriptor = observation.get("descriptor")
    if not isinstance(descriptor, list) or any(
        not isinstance(value, str) for value in descriptor
    ):
        raise ValueError("cached observation descriptor is invalid")
    return ImaginedTransition(
        action=request.action,
        observation=PredictedObservation(
            target_id=request.action.target_id,
            outcome=EvidenceOutcome(str(observation.get("outcome") or "")),
            descriptor=tuple(descriptor),
        ),
        belief_delta=CategoricalBeliefDelta(
            progress=ProgressChange(str(delta.get("progress") or "")),
            answerability_after=AnswerabilityState(
                str(delta.get("answerability_after") or "")
            ),
            frontier_change=FrontierChange(
                str(delta.get("frontier_change") or "")
            ),
            contradiction_change=ContradictionChange(
                str(delta.get("contradiction_change") or "")
            ),
            resolved_roles=_string_tuple(delta.get("resolved_roles"), "resolved_roles"),
            opened_roles=_string_tuple(delta.get("opened_roles"), "opened_roles"),
            relation_updates=_string_tuple(
                delta.get("relation_updates"), "relation_updates"
            ),
        ),
    )


def _string_tuple(value: object, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"cached {field} is invalid")
    return tuple(value)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {key: _jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _checksum(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _contains_number(value: Any) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return False
    if isinstance(value, (int, float)):
        return True
    if isinstance(value, dict):
        return any(_contains_number(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_number(item) for item in value)
    return False
