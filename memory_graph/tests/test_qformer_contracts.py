from __future__ import annotations

import json

import numpy as np
import pytest

from steam_video_new.implicit_world_model.reasoning_v2.qformer import (
    SLOT_NAMES,
    FeatureRow,
    FourSlotFeatureStore,
    merge_feature_stores,
    write_feature_store,
)
from steam_video_new.implicit_world_model.reasoning_v2.qformer.splits import (
    video_disjoint_indices,
)


def _write_store(tmp_path, *, validity=None):
    rows = tuple(
        FeatureRow(index, f"node:{index}", f"video:{index // 2}", f"lineage:{index}")
        for index in range(4)
    )
    arrays = {
        name: np.full((4, index + 2), index + 1, dtype=np.float32)
        for index, name in enumerate(SLOT_NAMES)
    }
    if validity is None:
        validity = np.ones((4, 4), dtype=np.bool_)
    return write_feature_store(
        tmp_path,
        arrays=arrays,
        validity=validity,
        rows=rows,
        encoders={name: f"fixture/{name}" for name in SLOT_NAMES},
        source_contract="fixture-safe-pre-read/v1",
        boundary_audit_version="fixture-boundary/v1",
    )


def test_feature_store_round_trip_and_coverage(tmp_path) -> None:
    validity = np.asarray(
        [[1, 1, 1, 1], [1, 0, 1, 1], [1, 1, 0, 1], [1, 1, 1, 1]],
        dtype=np.bool_,
    )
    manifest = _write_store(tmp_path, validity=validity)
    store = FourSlotFeatureStore(manifest)
    assert len(store) == 4
    assert store.node("node:1")["validity"].tolist() == [True, False, True, True]
    assert store.coverage() == {
        "caption": 1.0,
        "entity_state": 0.75,
        "visual": 0.75,
        "time": 1.0,
    }


def test_feature_store_rejects_checksum_drift(tmp_path) -> None:
    manifest = _write_store(tmp_path)
    caption = tmp_path / "caption.npy"
    caption.write_bytes(caption.read_bytes() + b"drift")
    with pytest.raises(ValueError, match="checksum mismatch"):
        FourSlotFeatureStore(manifest)


def test_feature_store_rejects_all_missing_node(tmp_path) -> None:
    validity = np.ones((4, 4), dtype=np.bool_)
    validity[2] = False
    with pytest.raises(ValueError, match="no valid slots"):
        _write_store(tmp_path, validity=validity)


def test_manifest_does_not_contain_grounded_value_payload(tmp_path) -> None:
    manifest = _write_store(tmp_path)
    payload = json.loads(manifest.read_text())
    serialized = json.dumps(payload)
    assert "EvidenceValue" not in serialized
    assert "grounded private value" not in serialized


def test_video_split_is_disjoint_and_deterministic() -> None:
    videos = ("a", "a", "b", "b", "c", "d")
    first = video_disjoint_indices(videos, validation_fraction=0.25)
    second = video_disjoint_indices(videos, validation_fraction=0.25)
    assert first == second
    train, validation = first
    assert {videos[index] for index in train}.isdisjoint(
        {videos[index] for index in validation}
    )


def test_merge_feature_stores_reindexes_disjoint_compatible_rows(tmp_path) -> None:
    first = _write_store(tmp_path / "first")
    rows = tuple(
        FeatureRow(index, f"other-node:{index}", f"other-video:{index // 2}", f"other:{index}")
        for index in range(4)
    )
    arrays = {
        name: np.full((4, index + 2), 10 + index, dtype=np.float32)
        for index, name in enumerate(SLOT_NAMES)
    }
    second = write_feature_store(
        tmp_path / "second",
        arrays=arrays,
        validity=np.ones((4, 4), dtype=np.bool_),
        rows=rows,
        encoders={name: f"fixture/{name}" for name in SLOT_NAMES},
        source_contract="fixture-safe-pre-read/v1",
        boundary_audit_version="fixture-boundary/v1",
    )

    merged = FourSlotFeatureStore(
        merge_feature_stores([first, second], tmp_path / "merged")
    )

    assert len(merged) == 8
    assert [row.row_index for row in merged.manifest.rows] == list(range(8))
    assert merged.node("node:0")["features"]["caption"].tolist() == [1.0, 1.0]
    assert merged.node("other-node:0")["features"]["caption"].tolist() == [10.0, 10.0]


def test_merge_feature_stores_rejects_overlapping_videos(tmp_path) -> None:
    first = _write_store(tmp_path / "first")
    second = _write_store(tmp_path / "second")
    with pytest.raises(ValueError, match="overlapping videos"):
        merge_feature_stores([first, second], tmp_path / "merged")
