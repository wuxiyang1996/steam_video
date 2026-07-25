from __future__ import annotations

import pytest

from steam_video_new.implicit_world_model.full_graph_iwm.multi_trajectory_cohort import (
    _select_case_ids,
    _validate_caption_candidate_mode,
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
