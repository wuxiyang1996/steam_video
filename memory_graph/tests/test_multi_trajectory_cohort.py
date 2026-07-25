from __future__ import annotations

import json

import pytest

from steam_video_new.implicit_world_model.full_graph_iwm.multi_trajectory_cohort import (
    MULTI_ARMS,
    _complete,
    _select_case_ids,
    _validate_caption_candidate_mode,
)
from steam_video_new.implicit_world_model.full_graph_iwm.pilot_analysis import (
    _consistent_model_name,
)


def test_explicit_cohort_selection_preserves_frozen_gate_order() -> None:
    gate = {"runnable_case_ids": ["case:b", "case:a", "case:c"]}

    selected = _select_case_ids(gate, ["case:c", "case:b"], None)

    assert selected == ["case:b", "case:c"]


def test_explicit_cohort_selection_rejects_non_runnable_case() -> None:
    gate = {"runnable_case_ids": ["case:a"]}

    with pytest.raises(ValueError, match="not runnable"):
        _select_case_ids(gate, ["case:unknown"], None)


def test_explicit_cohort_selection_rejects_duplicates_and_empty_selection() -> None:
    gate = {"runnable_case_ids": ["case:a"]}

    with pytest.raises(ValueError, match="unique"):
        _select_case_ids(gate, ["case:a", "case:a"], None)
    with pytest.raises(ValueError, match="no runnable"):
        _select_case_ids({"runnable_case_ids": []}, [], None)


def test_caption_candidate_mode_must_match_frozen_gate() -> None:
    gate = {
        "graphs": [
            {
                "graph_available": True,
                "caption_candidate_overlay_loaded": False,
            }
        ]
    }

    _validate_caption_candidate_mode(gate, disabled=True)
    with pytest.raises(ValueError, match="does not match frozen gate"):
        _validate_caption_candidate_mode(gate, disabled=False)


def test_pilot_analysis_uses_artifact_model_and_rejects_mixed_models() -> None:
    assert _consistent_model_name([{"model": "openai/gpt-5-mini"}]) == (
        "openai/gpt-5-mini"
    )
    with pytest.raises(ValueError, match="do not declare a model"):
        _consistent_model_name([{}])
    with pytest.raises(ValueError, match="mixes model identities"):
        _consistent_model_name([{"model": "model:a"}, {"model": "model:b"}])


def test_completed_case_artifact_requires_one_exact_run_per_arm(tmp_path) -> None:
    path = tmp_path / "case.json"
    artifact = {
        "selected_case_ids": ["case:a"],
        "arms": list(MULTI_ARMS),
        "runs": [{"case_id": "case:a", "arm": arm} for arm in MULTI_ARMS],
        "errors": [],
    }
    path.write_text(json.dumps(artifact), encoding="utf-8")
    assert _complete(path) is True

    artifact["runs"][-1] = artifact["runs"][0]
    path.write_text(json.dumps(artifact), encoding="utf-8")
    assert _complete(path) is False

    path.write_text("not-json", encoding="utf-8")
    assert _complete(path) is False
