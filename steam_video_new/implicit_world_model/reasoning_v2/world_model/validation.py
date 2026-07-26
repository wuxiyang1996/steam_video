"""Runtime validation that prevents hypothesis-conditioned observations."""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence

from .contracts import (
    ObservationPrediction,
    ObservationRequest,
    observation_world_key,
)


def validate_observation_predictions(
    requests: Sequence[ObservationRequest],
    predictions: Sequence[ObservationPrediction],
) -> tuple[ObservationPrediction, ...]:
    if len(requests) != len(predictions):
        raise ValueError("observation prediction coverage mismatch")
    request_by_id = {row.request_id: row for row in requests}
    if len(request_by_id) != len(requests):
        raise ValueError("duplicate observation request IDs")
    prediction_by_id = {row.request_id: row for row in predictions}
    if set(prediction_by_id) != set(request_by_id):
        raise ValueError("observation prediction IDs mismatch")
    grouped: dict[tuple[object, ...], list[ObservationPrediction]] = defaultdict(list)
    for request in requests:
        prediction = prediction_by_id[request.request_id]
        if prediction.action_id != request.action.action_id:
            raise ValueError("observation model changed the legal action")
        if prediction.target_id != request.target_address.node_id:
            raise ValueError("observation model changed the target")
        # Equal world state + action must have one observation irrespective of
        # which reasoning path requested it. Belief intentionally excludes an
        # answer hypothesis.
        key = observation_world_key(request)
        grouped[key].append(prediction)
    for rows in grouped.values():
        descriptors = {row.descriptor for row in rows}
        if len(descriptors) != 1:
            raise ValueError("observation prediction depends on reasoning hypothesis")
    return tuple(prediction_by_id[row.request_id] for row in requests)
