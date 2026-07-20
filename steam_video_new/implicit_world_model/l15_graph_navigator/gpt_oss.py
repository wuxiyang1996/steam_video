"""Categorical-only GPT-OSS adapters for the implicit WM and hop planner."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import runpy
from typing import Any
from urllib import error, request

from memory_graph.navigation import GraphReadAction, NavigationActionType
from memory_graph.types import CausalTemporalOverlay

from .contracts import (
    Answerability,
    BeliefDeltaDescriptor,
    BeliefSnapshot,
    ContradictionChange,
    EvidenceRole,
    FrontierChange,
    HypothesisDisposition,
    HypothesisUpdate,
    ObservationDescriptor,
    PathChange,
    PairwisePreference,
    PredictedTransition,
    PreferenceLabel,
    TrajectoryPrediction,
    UncertaintyChange,
    RecoveryStatus,
)
from .world_model import _ACTION_ROLE


DEFAULT_GPT_OSS_MODEL = "openai/gpt-oss-120b"
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"


class OpenAICompatibleCategoricalClient:
    """Small strict client; numeric model outputs are rejected."""

    def __init__(
        self,
        *,
        api_base: str,
        model: str = DEFAULT_GPT_OSS_MODEL,
        api_key: str | None = None,
        timeout_s: int = 180,
        max_tokens: int = 1600,
        reasoning_effort: str = "low",
    ) -> None:
        if not api_base:
            raise ValueError("api_base is required for GPT-OSS reasoning")
        self.endpoint = _chat_completions_endpoint(api_base)
        self.model = model
        self.api_key = api_key
        self.timeout_s = timeout_s
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if reasoning_effort not in {"low", "medium", "high"}:
            raise ValueError("reasoning_effort must be low, medium, or high")
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
        self.last_response_audit: dict[str, Any] = {}
        self.response_audits: list[dict[str, Any]] = []

    @classmethod
    def from_environment(
        cls,
        *,
        api_base: str | None = None,
        model: str = DEFAULT_GPT_OSS_MODEL,
        api_key_env: str = "OPENAI_API_KEY",
        timeout_s: int = 180,
        max_tokens: int = 1600,
        reasoning_effort: str = "low",
    ) -> OpenAICompatibleCategoricalClient:
        resolved_base = api_base or os.environ.get("OPENAI_BASE_URL")
        if not resolved_base:
            raise ValueError(
                "GPT-OSS backend requires --reasoning-api-base or OPENAI_BASE_URL"
            )
        return cls(
            api_base=resolved_base,
            model=model,
            api_key=os.environ.get(api_key_env),
            timeout_s=timeout_s,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
        )

    @classmethod
    def from_openrouter_keys_file(
        cls,
        keys_py_path: str | Path,
        *,
        api_base: str = OPENROUTER_API_BASE,
        model: str = DEFAULT_GPT_OSS_MODEL,
        timeout_s: int = 180,
        max_tokens: int = 1600,
        reasoning_effort: str = "low",
    ) -> OpenAICompatibleCategoricalClient:
        path = Path(keys_py_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"OpenRouter keys file does not exist: {path}")
        namespace = runpy.run_path(str(path))
        api_key = namespace.get("OPENROUTER_API_KEY")
        if not isinstance(api_key, str) or not api_key.strip():
            raise ValueError("keys.py must define a non-empty OPENROUTER_API_KEY")
        return cls(
            api_base=api_base,
            model=model,
            api_key=api_key,
            timeout_s=timeout_s,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
        )

    def complete_json(self, *, task: str, payload: dict[str, Any]) -> dict[str, Any]:
        body = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a multi-hop video-graph reasoning component. "
                        "Return exactly one JSON object. Use categorical labels only. "
                        "Never output a reward, score, confidence, probability, utility, "
                        "ranking number, or any other numeric value. Never treat an imagined "
                        "observation as acquired evidence."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {"task": task, "input": payload},
                        ensure_ascii=False,
                        allow_nan=False,
                    ),
                },
            ],
            "temperature": 0,
            "max_tokens": self.max_tokens,
            "response_format": {"type": "json_object"},
            "reasoning": {"effort": self.reasoning_effort},
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        api_request = request.Request(
            self.endpoint,
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with request.urlopen(api_request, timeout=self.timeout_s) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
        except (error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"GPT-OSS categorical request failed: {exc}") from exc
        try:
            choice = response_payload["choices"][0]
            content = choice["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("GPT-OSS response has no chat-completion content") from exc
        usage = response_payload.get("usage") or {}
        self.last_response_audit = {
            "finish_reason": choice.get("finish_reason"),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
        }
        self.response_audits.append(dict(self.last_response_audit))
        try:
            result = _parse_json_object(str(content))
        except ValueError as exc:
            raise ValueError(
                "GPT-OSS must return one complete JSON object; "
                f"finish_reason={choice.get('finish_reason')}"
            ) from exc
        _reject_numeric_output(result)
        return result


class GPTOSSObservationBeliefModel:
    """Predict categorical observation and belief-delta descriptors only."""

    def __init__(self, client: OpenAICompatibleCategoricalClient) -> None:
        self.client = client

    def predict(
        self,
        belief: BeliefSnapshot,
        action: GraphReadAction,
        overlay: CausalTemporalOverlay,
    ) -> PredictedTransition:
        if action.action_type is NavigationActionType.STOP:
            return PredictedTransition(
                action=action,
                observation=ObservationDescriptor(
                    role=EvidenceRole.NONE,
                    target_ids=(),
                    node_kind="none",
                    predicted_only=True,
                ),
                belief_delta=BeliefDeltaDescriptor(
                    uncertainty_change=UncertaintyChange.UNCHANGED,
                    answerability_after=belief.answerability,
                    predicted_only=True,
                ),
            )
        payload = {
            "query": belief.question,
            "belief": _categorical_belief_payload(belief),
            "reasoning_hop": _action_payload(action),
            "local_nodes": [
                {
                    "node_id": node.node_id,
                    "node_kind": node.node_type,
                    "description": node.text,
                    "evidence_refs": list(node.source_segments),
                }
                for node in overlay.atomic_events + overlay.l1_observations
            ],
            "local_relations": [
                {
                    "edge_id": edge.edge_id,
                    "src": edge.src,
                    "dst": edge.dst,
                    "relation_types": sorted(edge.relation_probabilities),
                }
                for edge in overlay.relations + overlay.l1_structural_relations
            ],
            "allowed_output": {
                "node_kind": "categorical string",
                "resolved_roles": "subset of missing_roles",
                "relation_updates": (
                    "copy only exact edge_id strings from local_relations; "
                    "use an empty list when no edge changes"
                ),
                "hypothesis_updates": (
                    "list of objects with edge_id copied from touched local_relations "
                    "and disposition in accepted/rejected/unresolved"
                ),
                "contradiction_updates": "categorical identifiers only",
                "frontier_change": [value.value for value in FrontierChange],
                "contradiction_change": [
                    value.value for value in ContradictionChange
                ],
                "path_change": [value.value for value in PathChange],
                "recovery_status": [value.value for value in RecoveryStatus],
                "uncertainty_change": [value.value for value in UncertaintyChange],
                "answerability_after": [value.value for value in Answerability],
            },
        }
        result = self.client.complete_json(
            task=(
                "Predict the descriptor that may follow this reasoning hop. "
                "The prediction is imagined and must not be cited as evidence."
            ),
            payload=payload,
        )
        missing = set(belief.missing_roles)
        allowed_edge_ids = {
            edge.edge_id
            for edge in overlay.relations + overlay.l1_structural_relations
            if action.source_id in {edge.src, edge.dst}
            and any(target in {edge.src, edge.dst} for target in action.target_ids)
        }
        resolved = _string_tuple(result.get("resolved_roles"), "resolved_roles")
        updates = _string_tuple(result.get("relation_updates"), "relation_updates")
        hypothesis_updates = _hypothesis_updates(result.get("hypothesis_updates"))
        if not set(resolved) <= missing:
            raise ValueError("GPT-OSS resolved_roles must be a subset of missing_roles")
        if not set(updates) <= allowed_edge_ids:
            raise ValueError(
                "GPT-OSS relation_updates must reference edges touched by the hop"
            )
        if not {update.edge_id for update in hypothesis_updates} <= allowed_edge_ids:
            raise ValueError(
                "GPT-OSS hypothesis_updates must reference edges touched by the hop"
            )
        expected_role = _ACTION_ROLE[action.action_type]
        return PredictedTransition(
            action=action,
            observation=ObservationDescriptor(
                role=expected_role,
                target_ids=action.target_ids,
                node_kind=str(result.get("node_kind") or "unknown"),
                predicted_only=True,
            ),
            belief_delta=BeliefDeltaDescriptor(
                resolved_roles=resolved,
                relation_updates=updates,
                hypothesis_updates=hypothesis_updates,
                contradiction_updates=_string_tuple(
                    result.get("contradiction_updates"),
                    "contradiction_updates",
                ),
                frontier_change=FrontierChange(
                    str(result.get("frontier_change") or "unchanged")
                ),
                contradiction_change=ContradictionChange(
                    str(result.get("contradiction_change") or "unchanged")
                ),
                path_change=PathChange(
                    str(result.get("path_change") or "unchanged")
                ),
                recovery_status=RecoveryStatus(
                    str(result.get("recovery_status") or "unchanged")
                ),
                uncertainty_change=UncertaintyChange(
                    str(result.get("uncertainty_change") or "unchanged")
                ),
                answerability_after=Answerability(
                    str(result.get("answerability_after") or "not_ready")
                ),
                predicted_only=True,
            ),
        )


class GPTOSSTrajectoryPreferenceModel:
    """Compare two complete candidate reasoning trajectories categorically."""

    def __init__(self, client: OpenAICompatibleCategoricalClient) -> None:
        self.client = client

    def compare(
        self,
        left: TrajectoryPrediction,
        right: TrajectoryPrediction,
        belief: BeliefSnapshot,
    ) -> PairwisePreference:
        result = self.client.complete_json(
            task=(
                "Compare the two candidate multi-hop reasoning trajectories. "
                "Prefer evidence-complete progress under the current belief."
            ),
            payload={
                "query": belief.question,
                "belief": _anonymous_preference_belief_payload(belief),
                "left": _trajectory_payload(left),
                "right": _trajectory_payload(right),
                "allowed_labels": [value.value for value in PreferenceLabel],
                "required_output": {
                    "label": "allowed label",
                    "rationale": "short categorical reason",
                },
            },
        )
        label = result.get("label")
        if not isinstance(label, str):
            raise ValueError(
                "GPT-OSS preference output must contain categorical label"
            )
        return PairwisePreference(
            left_id=left.trajectory_id,
            right_id=right.trajectory_id,
            label=PreferenceLabel(label),
            rationale=str(result.get("rationale") or "categorical comparison"),
        )


def _categorical_belief_payload(belief: BeliefSnapshot) -> dict[str, Any]:
    accepted: list[str] = []
    rejected: list[str] = []
    unresolved: list[str] = []
    for state in belief.relation_states:
        category = state.grounding.value
        item = f"{state.edge_id}:{category}"
        if category == "verified":
            accepted.append(item)
        elif category == "contradicted":
            rejected.append(item)
        else:
            unresolved.append(item)
    return {
        "answerability": belief.answerability.value,
        "uncertainty": belief.uncertainty.value,
        "missing_roles": list(belief.missing_roles),
        "contradictions": list(belief.contradictions),
        "accepted_hypotheses": accepted,
        "rejected_hypotheses": rejected,
        "unresolved_hypotheses": unresolved,
        "acquired_evidence_refs": list(belief.acquired_evidence),
    }


def _action_payload(action: GraphReadAction) -> dict[str, Any]:
    return {
        "operator": action.action_type.value,
        "source_id": action.source_id,
        "target_ids": list(action.target_ids),
        "relation": action.relation,
        "rationale": action.rationale,
    }


def _anonymous_preference_belief_payload(
    belief: BeliefSnapshot,
) -> dict[str, Any]:
    """Preserve task state while withholding graph-construction identifiers."""

    dispositions = [state.grounding.value for state in belief.relation_states]
    return {
        "answerability": belief.answerability.value,
        "uncertainty": belief.uncertainty.value,
        "missing_roles": list(belief.missing_roles),
        "contradiction_presence": (
            "present" if belief.contradictions else "absent"
        ),
        "hypothesis_dispositions": dispositions,
        "evidence_presence": (
            "present" if belief.acquired_evidence else "absent"
        ),
    }


def _trajectory_payload(trajectory: TrajectoryPrediction) -> dict[str, Any]:
    """Expose only anonymous imagined outcomes to the preference model.

    Action/operator names, graph identifiers, and free-text rationales are
    deliberately withheld so the comparator cannot select a hop by a lexical
    or graph-construction shortcut.
    """

    return {
        "hops": [
            {
                "observation_role": transition.observation.role.value,
                "observation_kind": transition.observation.node_kind,
                "resolved_roles": list(transition.belief_delta.resolved_roles),
                "hypothesis_dispositions": [
                    update.disposition.value
                    for update in transition.belief_delta.hypothesis_updates
                ],
                "frontier_change": transition.belief_delta.frontier_change.value,
                "contradiction_change": (
                    transition.belief_delta.contradiction_change.value
                ),
                "path_change": transition.belief_delta.path_change.value,
                "recovery_status": transition.belief_delta.recovery_status.value,
                "uncertainty_change": transition.belief_delta.uncertainty_change.value,
                "answerability_after": transition.belief_delta.answerability_after.value,
            }
            for transition in trajectory.transitions
        ],
    }


def _chat_completions_endpoint(api_base: str) -> str:
    base = api_base.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    if base.endswith("/v1"):
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"


def _parse_json_object(content: str) -> dict[str, Any]:
    stripped = content.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise ValueError("GPT-OSS must return one valid JSON object") from exc
    if not isinstance(payload, dict):
        raise ValueError("GPT-OSS categorical output must be a JSON object")
    return payload


def _reject_numeric_output(value: Any, *, path: str = "output") -> None:
    if isinstance(value, bool):
        return
    if isinstance(value, (int, float)):
        raise ValueError(f"GPT-OSS emitted a forbidden numeric value at {path}")
    if isinstance(value, dict):
        for key, child in value.items():
            _reject_numeric_output(child, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_numeric_output(child, path=f"{path}[{index}]")


def _string_tuple(value: Any, name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or not all(
        isinstance(item, str) for item in value
    ):
        raise ValueError(f"GPT-OSS {name} must be a list of strings")
    return tuple(dict.fromkeys(value))


def _hypothesis_updates(value: Any) -> tuple[HypothesisUpdate, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValueError("GPT-OSS hypothesis_updates must be a list of objects")
    updates: list[HypothesisUpdate] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, dict) or set(item) != {"edge_id", "disposition"}:
            raise ValueError(
                "GPT-OSS hypothesis_updates entries require edge_id and disposition"
            )
        edge_id = item["edge_id"]
        disposition = item["disposition"]
        if not isinstance(edge_id, str) or not isinstance(disposition, str):
            raise ValueError(
                "GPT-OSS hypothesis_updates entries must contain strings"
            )
        if edge_id in seen:
            raise ValueError("GPT-OSS hypothesis_updates contain duplicate edge_id")
        seen.add(edge_id)
        updates.append(
            HypothesisUpdate(
                edge_id=edge_id,
                disposition=HypothesisDisposition(disposition),
            )
        )
    return tuple(updates)
