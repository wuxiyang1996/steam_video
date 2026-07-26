from steam_video_new.implicit_world_model.full_graph_iwm.train_collection import (
    build_question_independent_graph_inputs,
    build_train_collection_manifest,
)


def test_train_collection_is_video_disjoint_and_answer_blind() -> None:
    dataset = {
        "dataset_id": "dataset:test",
        "cases": [
            {
                "case_id": f"case:{index}",
                "video_id": f"video:{index}",
                "video_ref": f"video:{index}.mp4",
                "video_duration_s": 10.0,
                "split": "train",
                "source": {"qid": str(index)},
                "planner_input": {"question": "hidden from selection"},
            }
            for index in range(40)
        ]
        + [
            {
                "case_id": "case:test-split",
                "video_id": "video:test-split",
                "video_ref": "test.mp4",
                "split": "test",
            }
        ],
    }

    manifest = build_train_collection_manifest(
        dataset, collection_id="collection:test", case_count=40
    )

    assert len(manifest["cases"]) == 40
    assert len({row["video_id"] for row in manifest["cases"]}) == 40
    assert {row["split"] for row in manifest["cases"]} == {"train"}
    assert manifest["selection_policy"]["question_text_used_for_selection"] is False
    assert manifest["selection_policy"]["hidden_answer_used_for_selection"] is False
    assert manifest["training_ready"] is False


def test_graph_worker_selection_is_separate_from_case_protocol(tmp_path) -> None:
    video = tmp_path / "cg_videos" / "video:0.mp4"
    video.parent.mkdir()
    video.write_bytes(b"video")
    collection = {
        "collection_id": "collection:test",
        "source_dataset_id": "dataset:test",
        "cases": [
            {
                "case_id": "case:0",
                "video_id": "video:0",
                "video_ref": "cg_videos/video:0.mp4",
                "video_duration_s": 10.0,
                "split": "train",
            }
        ],
    }
    graph_manifest = {
        "videos": [
            {
                "video_id": "video:0",
                "video_ref": "cg_videos/video:0.mp4",
                "current_candidate_source": "unavailable",
            }
        ]
    }

    selection, protocol = build_question_independent_graph_inputs(
        collection, graph_manifest, dataset_root=tmp_path
    )

    assert selection["selection_uses_question_or_gt"] is False
    assert "case_id" not in selection["videos"][0]
    assert protocol["cases"] == [
        {"case_id": "case:0", "video_id": "video:0", "split": "train"}
    ]
    assert protocol["graph_builder_receives_case_protocol"] is False
