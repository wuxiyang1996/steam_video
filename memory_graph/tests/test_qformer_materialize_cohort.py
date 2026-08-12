from __future__ import annotations

from steam_video_new.implicit_world_model.reasoning_v2.qformer.materialize_cohort import (
    _entity_state_text,
    _time_values,
)


def test_entity_state_serialization_is_bounded_to_node_metadata() -> None:
    node = {
        "metadata": {
            "participants": [
                {
                    "role": "actor",
                    "entity_type": "person",
                    "surface": "woman",
                    "visual_signature": "red coat",
                }
            ],
            "states": [{"attribute": "door", "value": "open"}],
        }
    }
    text = _entity_state_text(node)
    assert "actor | person | woman | red coat" in text
    assert "state=door:open" in text


def test_time_values_are_normalized_and_ordered() -> None:
    values = _time_values(
        {
            "time_span": {"start_s": 2.0, "end_s": 6.0},
            "_video_duration_s": 10.0,
        }
    )
    assert values == [0.2, 0.6, 0.4, 0.4]

