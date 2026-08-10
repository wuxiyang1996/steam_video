import json

import pytest

from steam_video_new.implicit_world_model.reasoning_v2.qformer.materialize_threebench import (
    _time_values,
    load_clip_rows,
)


def test_clip_rows_are_question_independent_and_collision_free(tmp_path):
    path = tmp_path / "clips.jsonl"
    rows = [
        {
            "dataset": "ovo_bench",
            "video_id": "v1",
            "clip_id": "clip:0",
            "video_path": "/data/v1.mp4",
            "start_s": 0,
            "end_s": 30,
            "text": "bounded caption",
            "question": "must be ignored",
            "answer": "must be ignored",
        },
        {
            "dataset": "ovo_bench",
            "video_id": "v1",
            "clip_id": "clip:1",
            "video_path": "/data/v1.mp4",
            "start_s": 30,
            "end_s": 60,
            "text": "another caption",
        },
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    loaded = load_clip_rows([path])
    assert len(loaded) == 2
    assert "question" not in loaded[0]
    assert "answer" not in loaded[0]
    assert loaded[0]["node_id"] != loaded[1]["node_id"]
    assert _time_values(loaded[0]) == pytest.approx([0.0, 0.5, 0.5, 0.25])


def test_reused_streaming_video_ids_do_not_collapse_distinct_paths(tmp_path):
    path = tmp_path / "clips.jsonl"
    rows = [
        {
            "dataset": "streaming_bench",
            "video_id": "reused",
            "clip_id": "clip:0",
            "video_path": f"/data/video-{index}.mp4",
            "start_s": 0,
            "end_s": 30,
            "text": "caption",
        }
        for index in range(2)
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    loaded = load_clip_rows([path])
    assert len(loaded) == 2
    assert loaded[0]["node_id"] != loaded[1]["node_id"]
    assert loaded[0]["video_id"] != loaded[1]["video_id"]
