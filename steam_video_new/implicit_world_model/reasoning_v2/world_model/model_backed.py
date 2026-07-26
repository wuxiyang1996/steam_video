"""Strict categorical model adapters for reasoning-v2 world-model heads."""

from __future__ import annotations

from typing import Any, Mapping, Protocol, Sequence

from .contracts import (
    Answerability,
    BeliefEffect,
    BeliefState,
    ContradictionChange,
    FrontierChange,
    GroundedBeliefEffect,
    HypothesisEffectRequest,
    ObservationDescriptor,
    ObservationKind,
    ObservationPrediction,
    ObservationRequest,
    Progress,
    RealObservation,
)


class CategoricalJSONClient(Protocol):
    model: str

    def complete_json(
        self, *, task: str, payload: dict[str, Any]
    ) -> dict[str, Any]: ...


class ModelBackedObservationWorldModel:
    """Predict one physical observation descriptor per target-bound action."""

    def __init__(
        self,
        client: CategoricalJSONClient,
        *,
        batch_size: int = 8,
    ) -> None:
        if batch_size < 1:
            raise ValueError("observation model batch_size must be positive")
        self.client = client
        self.model_name = str(client.model)
        self.batch_size = batch_size

    def predict_batch(
        self,
        requests: Sequence[ObservationRequest],
    ) -> tuple[ObservationPrediction, ...]:
        if not requests:
            return ()
        if len(requests) > self.batch_size:
            return tuple(
                prediction
                for start in range(0, len(requests), self.batch_size)
                for prediction in self.predict_batch(
                    requests[start : start + self.batch_size]
                )
            )
        aliases = tuple(f"request_{index}" for index in range(len(requests)))
        task = (
            "Predict the imagined physical observation descriptor for every "
            "target-bound reasoning action. The same physical action has one "
            "world outcome independent of answer hypothesis. Do not claim the "
            "prediction is observed evidence."
        )
        payload = {
            "requests": [
                _observation_request_payload(row, alias)
                for row, alias in zip(requests, aliases, strict=True)
            ],
            "required_output_schema": {
                "only_top_level_key": "predictions",
                "each_prediction_must_copy": "request_id",
            },
            "allowed_output": {
                "kind": [value.value for value in ObservationKind],
                "event_family": "categorical string",
                "entity_facts": "categorical string list",
                "state_facts": "categorical string list",
                "state_delta": "categorical string list",
                "rationale": "short categorical explanation",
            },
        }
        request_payload = payload
        for attempt in range(3):
            result = self.client.complete_json(
                task=(
                    task
                    if attempt == 0
                    else "Repair the observation prediction JSON to exactly match required_output_schema."
                ),
                payload=request_payload,
            )
            try:
                rows = _object_list(result.get("predictions"), "predictions")
                by_id = _rows_by_expected_ids(rows, aliases, "request_id")
                return tuple(
                    _observation_prediction(request, by_id[alias])
                    for request, alias in zip(requests, aliases, strict=True)
                )
            except ValueError as exc:
                if attempt == 2:
                    raise
                request_payload = _repair_payload(payload, result, exc)
        raise RuntimeError("unreachable observation schema repair")


class ModelBackedHypothesisEffectModel:
    """Interpret one imagined observation separately under each hypothesis."""

    def __init__(
        self,
        client: CategoricalJSONClient,
        *,
        batch_size: int = 8,
    ) -> None:
        if batch_size < 1:
            raise ValueError("hypothesis-effect batch_size must be positive")
        self.client = client
        self.model_name = str(client.model)
        self.batch_size = batch_size

    def predict_effect_batch(
        self,
        requests: Sequence[HypothesisEffectRequest],
    ) -> tuple[BeliefEffect, ...]:
        if not requests:
            return ()
        if len(requests) > self.batch_size:
            return tuple(
                effect
                for start in range(0, len(requests), self.batch_size)
                for effect in self.predict_effect_batch(
                    requests[start : start + self.batch_size]
                )
            )
        aliases = tuple(f"request_{index}" for index in range(len(requests)))
        task = (
            "Predict how each imagined observation would change its specified "
            "reasoning hypothesis and belief. Outputs are hypothesis-conditioned "
            "categorical effects, never rewards, scores, or acquired evidence."
        )
        payload = {
            "requests": [
                _effect_request_payload(row, alias)
                for row, alias in zip(requests, aliases, strict=True)
            ],
            "required_output_schema": {
                "only_top_level_key": "effects",
                "each_effect_must_copy": "request_id",
            },
            "allowed_output": {
                "progress": [value.value for value in Progress],
                "answerability_after": [value.value for value in Answerability],
                "contradiction_change": [value.value for value in ContradictionChange],
                "frontier_change": [value.value for value in FrontierChange],
                "resolved_roles": "subset of request missing_roles",
                "opened_roles": "subset of request required_roles",
                "rationale": "short categorical explanation",
            },
        }
        request_payload = payload
        for attempt in range(3):
            result = self.client.complete_json(
                task=(
                    task
                    if attempt == 0
                    else "Repair the hypothesis-effect JSON to exactly match required_output_schema and allowed_output."
                ),
                payload=request_payload,
            )
            try:
                rows = _object_list(result.get("effects"), "effects")
                by_id = _rows_by_expected_ids(rows, aliases, "request_id")
                return tuple(
                    _belief_effect(request, by_id[alias])
                    for request, alias in zip(requests, aliases, strict=True)
                )
            except ValueError as exc:
                if attempt == 2:
                    raise
                request_payload = _repair_payload(payload, result, exc)
        raise RuntimeError("unreachable hypothesis-effect schema repair")


class ModelBackedRealEffectCorrector:
    """Interpret one executed grounded value under every persistent hypothesis."""

    def __init__(self, client: CategoricalJSONClient) -> None:
        self.client = client
        self.model_name = str(client.model)

    def correct_batch(
        self,
        *,
        path_hypotheses: Mapping[str, str],
        beliefs: Mapping[str, BeliefState],
        acquired_ids_before: tuple[str, ...],
        observation: RealObservation,
    ) -> tuple[GroundedBeliefEffect, ...]:
        del acquired_ids_before
        if set(path_hypotheses) != set(beliefs):
            raise ValueError("real correction path coverage mismatch")
        aliases = {f"path_{index}": path_id for index, path_id in enumerate(beliefs)}
        task = (
            "Interpret the executed grounded evidence separately for every "
            "reasoning hypothesis. This is post-read correction, not imagination. "
            "Return categorical support/counterevidence/inconclusive judgments and "
            "only resolve roles directly grounded by this exact observation."
        )
        payload = {
            "real_observation": {
                "target_id": observation.target_id,
                "descriptor": observation.value.descriptor,
                "predicate": observation.value.predicate,
                "entities": [
                    {
                        "surface": entity.surface,
                        "role": entity.role,
                        "entity_type": entity.entity_type,
                    }
                    for entity in observation.value.entities
                ],
                "states": [
                    {
                        "attribute": state.attribute,
                        "value": state.value,
                        "polarity": state.polarity,
                    }
                    for state in observation.value.states
                ],
            },
            "paths": [
                {
                    "path_id": alias,
                    "hypothesis": path_hypotheses[path_id],
                    "question": belief.question,
                    "required_roles": list(belief.required_roles),
                    "missing_roles": list(belief.missing_roles),
                    "answerability": belief.answerability.value,
                }
                for alias, path_id in aliases.items()
                for belief in (beliefs[path_id],)
            ],
            "allowed_output": {
                "hypothesis_status": [
                    "support",
                    "counterevidence",
                    "inconclusive",
                ],
                "resolved_roles": "subset of that path missing_roles",
                "answerability_after": [value.value for value in Answerability],
                "rationale": "short direct-evidence explanation",
            },
            "required_output_schema": {
                "only_top_level_key": "corrections",
                "each_correction_must_copy": "path_id",
            },
        }
        request_payload = payload
        result = None
        by_id = None
        for attempt in range(3):
            result = self.client.complete_json(
                task=(
                    task
                    if attempt == 0
                    else "Repair the grounded correction JSON to exactly match required_output_schema and allowed_output."
                ),
                payload=request_payload,
            )
            try:
                rows = _object_list(result.get("corrections"), "corrections")
                by_id = _rows_by_expected_ids(rows, tuple(aliases), "path_id")
                break
            except ValueError as exc:
                if attempt == 2:
                    raise
                request_payload = _repair_payload(payload, result, exc)
        assert result is not None and by_id is not None
        effects = []
        alias_by_path = {path_id: alias for alias, path_id in aliases.items()}
        for path_id, belief in beliefs.items():
            row = by_id[alias_by_path[path_id]]
            status = str(row.get("hypothesis_status") or "")
            if status not in {"support", "counterevidence", "inconclusive"}:
                raise ValueError("invalid grounded hypothesis_status")
            resolved = _string_tuple(row.get("resolved_roles"), "resolved_roles")
            if not set(resolved).issubset(belief.missing_roles):
                raise ValueError("grounded resolved_roles must be missing")
            missing = tuple(
                role for role in belief.missing_roles if role not in resolved
            )
            contradictions = belief.contradictions
            if status == "counterevidence":
                contradictions = tuple(
                    dict.fromkeys(
                        (
                            *contradictions,
                            f"hypothesis_counterevidence:{path_hypotheses[path_id]}",
                        )
                    )
                )
            effects.append(
                GroundedBeliefEffect(
                    path_id=path_id,
                    belief_after=BeliefState(
                        question=belief.question,
                        required_roles=belief.required_roles,
                        missing_roles=missing,
                        contradictions=contradictions,
                        answerability=_enum(
                            Answerability,
                            row.get("answerability_after"),
                            "answerability_after",
                        ),
                        grounded_role_evidence=tuple(
                            dict.fromkeys(
                                (
                                    *belief.grounded_role_evidence,
                                    *(
                                        (role, observation.target_id)
                                        for role in resolved
                                    ),
                                )
                            )
                        ),
                    ),
                    direct_same_target=True,
                    verified=status != "inconclusive",
                    rationale=str(row.get("rationale") or ""),
                )
            )
        return tuple(effects)


def _observation_request_payload(
    request: ObservationRequest,
    external_request_id: str,
) -> dict[str, Any]:
    return {
        "request_id": external_request_id,
        "question": request.belief.question,
        "action": {
            "action_id": request.action.action_id,
            "kind": request.action.kind.value,
            "source_id": request.action.source_id,
            "target_id": request.action.target_id,
            "proposal_kind": (
                request.action.proposal_kind.value
                if request.action.proposal_kind is not None
                else None
            ),
        },
        "target_address": {
            "node_id": request.target_address.node_id,
            "event_family": request.target_address.event_family,
            "semantic_key": request.target_address.semantic_key,
            "structural_tags": list(request.target_address.structural_tags),
            "source_segments": list(request.target_address.source_segments),
            "embedding_available": request.target_address.embedding_ref is not None,
        },
        "acquired_evidence": [
            {
                "node_id": node_id,
                "descriptor": value.descriptor,
                "predicate": value.predicate,
                "entities": [entity.surface for entity in value.entities],
                "states": [
                    f"{state.attribute}:{state.value}:{state.polarity}"
                    for state in value.states
                ],
                "state_delta": (
                    None
                    if value.state_delta is None
                    else (
                        f"{value.state_delta.attribute}:"
                        f"{value.state_delta.before}->{value.state_delta.after}"
                    )
                ),
            }
            for node_id, value in request.context.acquired_values
        ],
        "imagined_prefix": [
            {
                "target_id": target_id,
                "kind": descriptor.kind.value,
                "event_family": descriptor.event_family,
                "entity_facts": list(descriptor.entity_facts),
                "state_facts": list(descriptor.state_facts),
                "state_delta": list(descriptor.state_delta),
            }
            for target_id, descriptor in request.context.imagined_prefix
        ],
    }


def _effect_request_payload(
    request: HypothesisEffectRequest,
    external_request_id: str,
) -> dict[str, Any]:
    return {
        "request_id": external_request_id,
        "path_id": request.path_id,
        "hypothesis": request.hypothesis,
        "belief": {
            "question": request.belief.question,
            "required_roles": list(request.belief.required_roles),
            "missing_roles": list(request.belief.missing_roles),
            "contradictions": list(request.belief.contradictions),
            "answerability": request.belief.answerability.value,
            "grounded_role_evidence": [
                list(row) for row in request.belief.grounded_role_evidence
            ],
        },
        "imagined_observation": {
            "target_id": request.observation.target_id,
            "kind": request.observation.descriptor.kind.value,
            "event_family": request.observation.descriptor.event_family,
            "entity_facts": list(request.observation.descriptor.entity_facts),
            "state_facts": list(request.observation.descriptor.state_facts),
            "state_delta": list(request.observation.descriptor.state_delta),
        },
    }


def _observation_prediction(
    request: ObservationRequest,
    payload: dict[str, Any],
) -> ObservationPrediction:
    return ObservationPrediction(
        request_id=request.request_id,
        action_id=request.action.action_id,
        target_id=request.target_address.node_id,
        descriptor=ObservationDescriptor(
            kind=_enum(ObservationKind, payload.get("kind"), "kind"),
            event_family=_required_string(payload.get("event_family"), "event_family"),
            entity_facts=_string_tuple(payload.get("entity_facts"), "entity_facts"),
            state_facts=_string_tuple(payload.get("state_facts"), "state_facts"),
            state_delta=_string_tuple(payload.get("state_delta"), "state_delta"),
        ),
        rationale=str(payload.get("rationale") or ""),
    )


def _belief_effect(
    request: HypothesisEffectRequest,
    payload: dict[str, Any],
) -> BeliefEffect:
    resolved = _string_tuple(payload.get("resolved_roles"), "resolved_roles")
    opened = _string_tuple(payload.get("opened_roles"), "opened_roles")
    if not set(resolved).issubset(request.belief.missing_roles):
        raise ValueError("resolved_roles must be a subset of missing_roles")
    if not set(opened).issubset(request.belief.required_roles):
        raise ValueError("opened_roles must be a subset of required_roles")
    return BeliefEffect(
        request_id=request.request_id,
        progress=_enum(Progress, payload.get("progress"), "progress"),
        answerability_after=_enum(
            Answerability,
            payload.get("answerability_after"),
            "answerability_after",
        ),
        contradiction_change=_enum(
            ContradictionChange,
            payload.get("contradiction_change"),
            "contradiction_change",
        ),
        frontier_change=_enum(
            FrontierChange,
            payload.get("frontier_change"),
            "frontier_change",
        ),
        resolved_roles=resolved,
        opened_roles=opened,
        rationale=str(payload.get("rationale") or ""),
    )


def _rows_by_expected_ids(
    rows: list[dict[str, Any]],
    expected: tuple[str, ...],
    id_field: str,
) -> dict[str, dict[str, Any]]:
    identifiers = [str(row.get(id_field) or "") for row in rows]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError(f"model output has duplicate {id_field}")
    if set(identifiers) != set(expected):
        raise ValueError(f"model output {id_field} coverage mismatch")
    return dict(zip(identifiers, rows, strict=True))


def _object_list(value: Any, field: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise ValueError(f"{field} must be a list of objects")
    return value


def _string_tuple(value: Any, field: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item.strip() for item in value
    ):
        raise ValueError(f"{field} must be a list of non-empty strings")
    return tuple(dict.fromkeys(value))


def _required_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _enum(enum_type: Any, value: Any, field: str) -> Any:
    try:
        return enum_type(str(value))
    except ValueError as exc:
        raise ValueError(f"invalid categorical {field}: {value}") from exc


def _repair_payload(
    payload: dict[str, Any],
    invalid_response: dict[str, Any],
    error: ValueError,
) -> dict[str, Any]:
    return {
        **payload,
        "repair_feedback": {
            "validation_error": str(error),
            "invalid_response": invalid_response,
            "instruction": (
                "Preserve categorical content but copy exact requested IDs and field "
                "names. Return no additional top-level fields."
            ),
        },
    }
