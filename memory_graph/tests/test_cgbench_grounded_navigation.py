from __future__ import annotations

from pathlib import Path

from steam_video_new.implicit_world_model.cgbench_grounded_navigation import (
    attach_descriptor_embeddings,
    build_ablation_manifest,
    build_blinded_review_packet,
    build_cgbench_navigation_dataset,
    ground_navigation_dataset,
    validate_cgbench_navigation_dataset,
)


def _row(video_id: str, qid: int, intervals: list[list[int]]) -> dict:
    return {
        "qid": qid,
        "video_uid": video_id,
        "question": "Which event resolves the question?",
        "answer": "the hidden answer",
        "choices": ["first choice", "the hidden answer", "third choice"],
        "right_answer": "B",
        "clue_intervals": intervals,
        "duration": 100,
    }


def test_builds_video_disjoint_leakage_safe_categorical_navigation_data(
    tmp_path: Path,
) -> None:
    for video_id in ("video-a", "video-b"):
        (tmp_path / f"{video_id}.mp4").write_bytes(b"container")
    dataset, hidden, report = build_cgbench_navigation_dataset(
        [
            _row("video-a", 1, [[10, 14], [40, 46]]),
            _row("video-a", 2, [[20, 23], [70, 75]]),
            _row("video-b", 1, [[5, 8], [50, 54]]),
        ],
        video_root=tmp_path,
        dataset_id="cgbench:test",
        duration_probe=lambda _: 100.0,
    )

    assert validate_cgbench_navigation_dataset(dataset, hidden) == []
    assert report["case_count"] == 3
    assert report["numeric_reward_present"] is False
    assert dataset["training_ready"] is False
    assert (
        len(
            {
                case["split"]
                for case in dataset["cases"]
                if case["video_id"] == "video-a"
            }
        )
        == 1
    )
    for case in dataset["cases"]:
        planner_text = str(case["planner_input"])
        assert "right_answer" not in planner_text
        assert "clue_intervals" not in planner_text
        assert ":candidate:" not in planner_text
        assert ":control:" not in planner_text
        comparison_text = str(case["trajectory_preference"])
        assert "terminal_evidence_coverage" not in comparison_text
        assert case["trajectory_preference"]["label"] in {"prefer_left", "prefer_right"}
        transitions = case["executed_transitions"]
        positive = [
            row
            for row in transitions
            if row["target"]["belief_delta"]["evidence_progress"]
            == "advances_required_clue_coverage"
        ]
        assert len(positive) == len(transitions) == 2
        assert "does_not_advance_required_clue_coverage" not in str(transitions)
        comparison = case["trajectory_preference"]
        preferred = (
            comparison["left"]
            if comparison["label"] == "prefer_left"
            else comparison["right"]
        )
        other = (
            comparison["right"]
            if comparison["label"] == "prefer_left"
            else comparison["left"]
        )
        assert len(preferred["action_ids"]) == len(other["action_ids"]) + 1
        assert set(other["action_ids"]) < set(preferred["action_ids"])
        assert all(
            row["target"]["observation_descriptor"]["embedding_ref"]["model"]
            == "Qwen/Qwen3-VL-Embedding-2B"
            for row in transitions
        )
    assert all("answer_key" in row for row in hidden["cases"])


def test_quarantines_clues_outside_real_video_duration(tmp_path: Path) -> None:
    (tmp_path / "broken.mp4").write_bytes(b"container")
    dataset, hidden, report = build_cgbench_navigation_dataset(
        [_row("broken", 1, [[10, 20], [90, 110]])],
        video_root=tmp_path,
        dataset_id="cgbench:quarantine",
        duration_probe=lambda _: 100.0,
    )

    assert dataset["cases"] == []
    assert hidden["cases"] == []
    assert report["quarantined_video_count"] == 1
    assert report["quarantined"][0]["reason"] == "clue_window_exceeds_video_duration"


def test_audited_source_integrity_exclusion_is_exact_and_checksummed(
    tmp_path: Path,
) -> None:
    for video_id in ("bad-video", "good-video"):
        (tmp_path / f"{video_id}.mp4").write_bytes(b"container")
    dataset, hidden, report = build_cgbench_navigation_dataset(
        [
            _row("bad-video", 7, [[10, 14], [40, 46]]),
            _row("good-video", 8, [[10, 14], [40, 46]]),
        ],
        video_root=tmp_path,
        dataset_id="cgbench:source-integrity",
        duration_probe=lambda _: 100.0,
        source_integrity_exclusions=(
            {
                "video_uid": "bad-video",
                "qid": "7",
                "reason": "audited source binding defect",
            },
        ),
    )

    assert [row["video_id"] for row in dataset["cases"]] == ["good-video"]
    assert report["source_integrity_exclusion_count"] == 1
    assert report["skipped"]["source_integrity_exclusion"] == 1
    assert len(dataset["cases"][0]["source"]["public_source_sha256"]) == 64
    assert len(hidden["cases"][0]["source_record_sha256"]) == 64


class _FakeVLM:
    model = "Qwen/Qwen3.5-9B"

    def perceive(self, prompt: str, *, image_urls=None, system: str = "") -> dict:
        return {
            "summary": "A person opens a visible door.",
            "visible_entities": ["person", "door"],
            "visible_actions": ["opens door"],
            "visible_states": ["door becomes open"],
            "readable_text": [],
            "question_relevance": "inconclusive",
            "relevance_reason": "The sampled view does not establish the answer.",
            "evidence_frames": [0],
        }


class _FakeEmbedding:
    model_name = "Qwen/Qwen3-VL-Embedding-2B"
    dimension = 2048

    def encode(self, texts, *, batch_size: int = 8):
        return [[1.0] + [0.0] * 2047 for _ in texts]


def test_grounding_embedding_review_and_ablation_contracts(
    tmp_path: Path, monkeypatch
) -> None:
    import steam_video_new.implicit_world_model.cgbench_grounded_navigation.grounding as grounding

    (tmp_path / "video-a.mp4").write_bytes(b"container")
    dataset, hidden, _ = build_cgbench_navigation_dataset(
        [_row("video-a", 1, [[10, 14], [40, 46]])],
        video_root=tmp_path,
        dataset_id="cgbench:ground",
        duration_probe=lambda _: 100.0,
    )
    monkeypatch.setattr(
        grounding,
        "_sample_frame_data_uris",
        lambda *args, **kwargs: (
            ["data:image/jpeg;base64,AA=="],
            [{"frame_index": 0, "time_s": 10.0, "purpose": "candidate_read"}],
        ),
    )
    grounded, report = ground_navigation_dataset(
        dataset, video_root=tmp_path, client=_FakeVLM()
    )
    assert report["grounded_transition_count"] == 2
    assert all(
        row["real_observation"]["descriptor_status"] == "grounded_qwen_vl_read"
        for row in grounded["cases"][0]["executed_transitions"]
    )
    embedded, manifest = attach_descriptor_embeddings(
        grounded, _FakeEmbedding(), output_path=tmp_path / "vectors.npy"
    )
    assert manifest["row_count"] == 2
    assert all(
        row["target"]["observation_descriptor"]["embedding_ref"]["status"]
        == "available"
        for row in embedded["cases"][0]["executed_transitions"]
    )
    from steam_video_new.implicit_world_model.cgbench_grounded_navigation.builder import (
        _checksum,
    )
    from steam_video_new.implicit_world_model.cgbench_grounded_navigation.validate_artifact import (
        validate_completed_artifact,
    )

    embedded_hidden = {**hidden, "dataset_sha256": _checksum(embedded)}
    validation = validate_completed_artifact(
        embedded, embedded_hidden, manifest, matrix_path=tmp_path / "vectors.npy"
    )
    assert validation["status"] == "pass"
    assert validation["embedding_shape"] == [2, 2048]

    ablation, key = build_ablation_manifest(embedded, hidden)
    assert [arm["arm"] for arm in ablation["cases"][0]["arms"]] == [
        "all_clues",
        "leave_one_clue_out",
        "shuffled_clue_order",
        "transition_prediction_shuffle",
        "cross_video_distractor",
    ]
    assert ablation["human_review_required"] is False
    assert "matched_control" not in str(ablation)
    assert "answer_key" not in str(ablation)
    assert key["cases"][0]["answer_key"] == "B"
    review, review_key = build_blinded_review_packet(embedded, hidden, sample_videos=1)
    assert review["items"] and "gt_interval_role" not in str(review)
    assert review["formal_gate"] is False
    assert {row["gt_interval_role"] for row in review_key["items"]} == {"clue"}


def test_invalid_qwen_relevance_label_is_not_promoted_to_supervision() -> None:
    from steam_video_new.implicit_world_model.cgbench_grounded_navigation.grounding import (
        _validated_descriptor,
    )

    descriptor = _validated_descriptor(
        {
            "summary": "The subtitle mentions an English exam.",
            "visible_entities": [],
            "visible_actions": [],
            "visible_states": [],
            "readable_text": ["English exam"],
            "question_relevance": "high",
            "relevance_reason": "It may contribute one item.",
            "evidence_frames": [0],
        },
        [{"frame_index": 0}],
    )
    assert descriptor["question_relevance"] == "inconclusive"
    assert descriptor["raw_question_relevance"] == "high"

    one_based = _validated_descriptor(
        {
            "summary": "A visible action occurs.",
            "visible_entities": [],
            "visible_actions": [],
            "visible_states": [],
            "readable_text": [],
            "question_relevance": "inconclusive",
            "relevance_reason": "The evidence is partial.",
            "evidence_frames": [1, 2],
        },
        [{"frame_index": 0}, {"frame_index": 1}],
    )
    assert one_based["evidence_frames"] == [0, 1]
    assert one_based["frame_index_status"] == "canonicalized_from_one_based"


def test_gt_ablation_distractor_is_evaluation_only_and_video_disjoint(
    tmp_path: Path,
) -> None:
    for video_id in ("video-a", "video-b"):
        (tmp_path / f"{video_id}.mp4").write_bytes(b"container")
    dataset, hidden, _ = build_cgbench_navigation_dataset(
        [
            _row("video-a", 1, [[10, 14], [40, 46]]),
            _row("video-b", 2, [[5, 9], [60, 66]]),
        ],
        video_root=tmp_path,
        dataset_id="cgbench:ablation",
        duration_probe=lambda _: 100.0,
    )
    ablation, _ = build_ablation_manifest(dataset, hidden)
    for case in ablation["cases"]:
        arms = {arm["arm"]: arm for arm in case["arms"]}
        distractor = arms["cross_video_distractor"]
        assert distractor["available"] is True
        by_id = {action["action_id"]: action for action in case["actions"]}
        assert all(
            by_id[action_id]["evaluation_only"] is True
            for action_id in distractor["action_ids"]
        )
        assert all(
            by_id[action_id]["source_video_id"] != case["video_id"]
            for action_id in distractor["action_ids"]
        )
        shuffled = arms["transition_prediction_shuffle"]
        assert shuffled["prediction_source_action_ids"] != shuffled["action_ids"]


def test_real_candidate_hops_keep_gt_alignment_hidden_and_nonmatches_unlabeled(
    tmp_path: Path,
) -> None:
    import json
    from steam_video_new.implicit_world_model.cgbench_grounded_navigation.l15_candidates import (
        build_closed_loop_protocol,
        build_l15_candidate_artifact,
        validate_l15_candidate_artifact,
    )
    from steam_video_new.implicit_world_model.cgbench_grounded_navigation.modality_coverage import (
        build_modality_coverage,
    )

    for video_id in ("video-a", "video-b"):
        (tmp_path / f"{video_id}.mp4").write_bytes(b"container")
    dataset, hidden, _ = build_cgbench_navigation_dataset(
        [
            _row("video-a", 1, [[10, 14], [40, 46]]),
            _row("video-b", 2, [[5, 9], [60, 66]]),
        ],
        video_root=tmp_path,
        dataset_id="cgbench:candidates",
        duration_probe=lambda _: 100.0,
    )
    graph_path = tmp_path / "video-a.graph.json"
    graph_path.write_text(
        json.dumps(
            {
                "video_id": "video-a",
                "nodes": [
                    {
                        "node_id": "node:hit",
                        "time_span": {"start_s": 11, "end_s": 12},
                        "text": "grounded graph evidence",
                        "node_type": "observation",
                    },
                    {
                        "node_id": "node:other",
                        "time_span": {"start_s": 80, "end_s": 82},
                        "text": "unlabeled graph evidence",
                        "node_type": "observation",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    subtitle_root = tmp_path / "subtitles"
    subtitle_root.mkdir()
    (subtitle_root / "video-b.srt").write_text(
        "1\n00:00:05,000 --> 00:00:06,000\nfirst clue text\n\n"
        "2\n00:00:80,000 --> 00:00:82,000\noutside annotation\n",
        encoding="utf-8",
    )
    public, alignment, report = build_l15_candidate_artifact(
        dataset, hidden, graph_paths=[graph_path], subtitle_root=subtitle_root
    )
    assert validate_l15_candidate_artifact(public, alignment) == []
    assert report["source_case_counts"] == {
        "persisted_l1_l15_graph": 1,
        "time_aligned_subtitle_l1_fallback": 1,
    }
    assert len(public["candidate_sets"]) == 2
    assert all("candidate_hops" not in case for case in public["cases"])
    assert "clue_intervals" not in str(public)
    nonmatches = [
        row
        for case in alignment["cases"]
        for row in case["hop_alignment"]
        if row["temporal_gt_relation"] == "no_overlap_with_annotated_clue_unlabeled"
    ]
    assert nonmatches and all(row["semantic_negative"] is False for row in nonmatches)
    protocol = build_closed_loop_protocol(public)
    assert [row["arm"] for row in protocol["arms"]] == [
        "world_model_guided",
        "no_world_model",
        "shuffled_world_model_prediction",
        "frozen_world_model",
        "immediate_effect_only",
        "oracle_clue_ceiling",
    ]
    summary, details = build_modality_coverage(
        dataset, hidden, subtitle_root=subtitle_root
    )
    assert summary["human_review_required"] is False
    assert summary["video_with_subtitle_count"] == 1
    assert all(row["direct_audio_observed"] is False for row in details["clues"])


def test_frozen_correlation_evaluation_uses_gt_only_after_graph_build(
    tmp_path: Path,
) -> None:
    import json
    from steam_video_new.implicit_world_model.cgbench_grounded_navigation.l15_graph_worker import (
        _shortest_path_hops,
        evaluate_frozen_l15_correlations,
    )

    videos = [("video-train", "train"), ("video-test", "test")]
    selection = {
        "dataset_id": "cgbench:correlation-audit",
        "observation_horizon_s": 120.0,
        "videos": [
            {"video_id": video_id, "split": split} for video_id, split in videos
        ],
    }
    dataset = {
        "dataset_id": "cgbench:correlation-audit",
        "cases": [
            {"case_id": f"case:{video_id}", "video_id": video_id, "split": split}
            for video_id, split in videos
        ],
    }
    hidden = {
        "cases": [
            {
                "case_id": f"case:{video_id}",
                "clue_intervals": [
                    {"start_s": 0.0, "end_s": 2.0},
                    {"start_s": 10.0, "end_s": 12.0},
                ],
            }
            for video_id, _ in videos
        ]
    }
    for video_id, _ in videos:
        sample_dir = tmp_path / video_id
        sample_dir.mkdir()
        graph = {
            "nodes": [
                {
                    "node_id": "l1:a",
                    "time_span": {"start_s": 0.0, "end_s": 2.0},
                },
                {
                    "node_id": "l1:b",
                    "time_span": {"start_s": 10.0, "end_s": 12.0},
                },
            ],
            "temporal_edges": [],
            "correlation_edges": [
                {"src": "l1:a", "dst": "l1:b", "edge_id": "corr:a:b"}
            ],
            "metadata": {
                "question_independent": True,
                "source_l1_fingerprint": f"source:{video_id}",
                "retained_l1_fingerprint": f"retained:{video_id}",
            },
        }
        audit = {
            "contains_question_or_answer": False,
            "pairs": [
                {
                    "src": "l1:a",
                    "dst": "l1:b",
                    "features": {"semantic_similarity": 0.8},
                }
            ],
        }
        (sample_dir / "l1_l15_navigation_graph.json").write_text(
            json.dumps(graph), encoding="utf-8"
        )
        (sample_dir / "l1_l15_correlation_pair_audit.json").write_text(
            json.dumps(audit), encoding="utf-8"
        )

    report, details = evaluate_frozen_l15_correlations(
        dataset,
        hidden,
        selection,
        graph_root=tmp_path,
        target_positive_coverage=1.0,
    )

    assert report["bridge_metrics"]["coverage"] == 1.0
    assert report["bridge_metrics"]["soft_correlation_coverage"] == 1.0
    assert report["admitted_edge_precision"] is None
    assert report["unmatched_pairs_treated_as_negative"] is False
    assert report["calibration"]["heldout_positive_coverage"] == 1.0
    assert report["calibration"]["application_status"] == (
        "diagnostic_only_not_applied_to_graph"
    )
    assert "GT clue intervals" in details["warning"]
    assert (
        _shortest_path_hops(
            {"a"},
            {"c"},
            {"a": {"b"}, "b": {"a", "c"}, "c": {"b"}},
            max_hops=2,
        )
        == 2
    )


def test_question_independent_l15_selection_and_frozen_coverage(tmp_path: Path) -> None:
    import json
    import numpy as np
    from steam_video_new.implicit_world_model.cgbench_grounded_navigation.l15_graph_worker import (
        _limit_selection,
        _select_video_shard,
        evaluate_frozen_graph_coverage,
        merge_extract_reports,
        select_smoke_videos,
    )

    videos = []
    counter = 0
    for split in ("train", "validation", "test"):
        for source in ("time_aligned_subtitle_l1_fallback", "unavailable"):
            for _ in range(2):
                video_id = f"video-{counter}"
                (tmp_path / f"{video_id}.mp4").write_bytes(b"container")
                videos.append(
                    {
                        "video_id": video_id,
                        "video_ref": f"{video_id}.mp4",
                        "current_candidate_source": source,
                        "cases": [{"case_id": f"case-{counter}", "split": split}],
                    }
                )
                counter += 1
    selection = select_smoke_videos(
        {"dataset_id": "cgbench:smoke", "videos": videos},
        dataset_root=tmp_path,
        duration_probe=lambda path: float(len(path.name)),
    )
    assert len(selection["videos"]) == 12
    assert selection["selection_uses_question_or_gt"] is False
    assert len(_limit_selection(selection, 8)["videos"]) == 8
    shards = [
        _select_video_shard(selection, video_limit=8, shard_index=index, num_shards=4)
        for index in range(4)
    ]
    assert [[row["video_id"] for row in shard["videos"]] for shard in shards] == [
        [selection["videos"][0]["video_id"], selection["videos"][4]["video_id"]],
        [selection["videos"][1]["video_id"], selection["videos"][5]["video_id"]],
        [selection["videos"][2]["video_id"], selection["videos"][6]["video_id"]],
        [selection["videos"][3]["video_id"], selection["videos"][7]["video_id"]],
    ]

    graph_root = tmp_path / "merge-graphs"
    report_root = graph_root / "shard_reports"
    report_root.mkdir(parents=True)
    for index, shard in enumerate(shards):
        samples = []
        for row in shard["videos"]:
            video_id = row["video_id"]
            sample_dir = graph_root / video_id
            sample_dir.mkdir(parents=True)
            (sample_dir / "video_l1.json").write_text("{}", encoding="utf-8")
            (sample_dir / "causal_temporal_overlay.json").write_text(
                json.dumps({"metadata": {"question_independent_contract": True}}),
                encoding="utf-8",
            )
            samples.append({"video_id": video_id, "status": "completed"})
        (report_root / f"build_report.shard_{index}_of_4.json").write_text(
            json.dumps(
                {
                    "schema_version": "steam-cgbench-question-independent-l15-build/v0.1",
                    "status": "completed",
                    "samples": samples,
                    "errors": [],
                }
            ),
            encoding="utf-8",
        )
    merged = merge_extract_reports(
        selection,
        graph_root=graph_root,
        shard_report_dir=report_root,
        num_shards=4,
        video_limit=8,
    )
    assert merged["status"] == "completed"
    assert merged["completed_video_count"] == 8
    assert [row["video_id"] for row in merged["samples"]] == [
        row["video_id"] for row in selection["videos"][:8]
    ]
    assert merged["parallel_extraction"]["num_shards"] == 4

    (tmp_path / "video-a.mp4").write_bytes(b"container")
    dataset, hidden, _ = build_cgbench_navigation_dataset(
        [_row("video-a", 1, [[10, 14], [40, 46]])],
        video_root=tmp_path,
        dataset_id="cgbench:frozen",
        duration_probe=lambda _: 100.0,
    )
    graph_root = tmp_path / "graphs"
    graph_dir = graph_root / "video-a"
    graph_dir.mkdir(parents=True)
    nodes = [
        {
            "node_id": "node:hit",
            "video_id": "video-a",
            "node_type": "observation",
            "time_span": {"start_s": 10, "end_s": 12},
            "text": "relevant event",
        },
        {
            "node_id": "node:other",
            "video_id": "video-a",
            "node_type": "observation",
            "time_span": {"start_s": 80, "end_s": 82},
            "text": "other event",
        },
    ]
    graph_path = graph_dir / "causal_temporal_overlay.json"
    graph_path.write_text(
        json.dumps(
            {
                "schema_version": "steam-causal-overlay/v0.2",
                "overlay_id": "overlay:a",
                "example_id": "example:a",
                "video_id": "video-a",
                "l1_observations": nodes,
                "atomic_events": [],
                "relations": [],
                "l1_structural_relations": [],
                "metadata": {"question_independent_contract": True},
            }
        ),
        encoding="utf-8",
    )
    matrix = np.zeros((2, 2048), dtype=np.float32)
    matrix[0, 0] = 1.0
    matrix[1, 1] = 1.0
    np.save(graph_dir / "node_embeddings.npy", matrix)
    (graph_dir / "node_embeddings.manifest.json").write_text(
        json.dumps(
            {
                "rows": [
                    {"row_index": 0, "node_id": "node:hit"},
                    {"row_index": 1, "node_id": "node:other"},
                ],
            }
        ),
        encoding="utf-8",
    )
    frozen_selection = {
        "observation_horizon_s": 100.0,
        "videos": [{"video_id": "video-a", "observation_horizon_s": 100.0}],
    }
    report, details = evaluate_frozen_graph_coverage(
        dataset,
        hidden,
        frozen_selection,
        graph_root=graph_root,
        query_provider=_FakeEmbedding(),
        top_ks=(1, 2),
    )
    assert report["graph_video_count"] == 1
    assert report["embedding_top_k_recall"]["1"]["covered"] == 1
    assert report["nonoverlap_candidates_are_semantic_negatives"] is False
    assert details["cases"][0]["clues"][0]["top_k_status"]["1"] == "covered"


def test_gpt56_audit_is_categorical_and_only_joins_hidden_labels_after_review() -> None:
    from steam_video_new.implicit_world_model.cgbench_grounded_navigation.gpt56_audit import (
        _validate_decision,
        inspect_audit_against_hidden_key,
    )

    decision = _validate_decision(
        "review:one",
        {
            "media_alignment": "accept",
            "descriptor_grounding": "accept",
            "question_relevance": "relevant",
            "evidence_kind": "readable_text",
            "rationale": "The subtitle contributes one requested subject.",
        },
    )
    audit = {"model": "openai/gpt-5.6-sol", "decisions": [decision]}
    report = inspect_audit_against_hidden_key(
        audit, {"items": [{"review_id": "review:one", "gt_interval_role": "control"}]}
    )
    assert report["role_by_relevance_counts"]["control"]["relevant"] == 1
    assert report["formal_eligible"] is False

    import pytest

    with pytest.raises(ValueError, match="numeric"):
        _validate_decision(
            "review:bad",
            {
                "media_alignment": "accept",
                "descriptor_grounding": "accept",
                "question_relevance": "relevant",
                "evidence_kind": "visual",
                "rationale": "supported",
                "confidence": 0.9,
            },
        )


def test_provisional_filter_rejects_contaminated_control() -> None:
    from steam_video_new.implicit_world_model.cgbench_grounded_navigation.filter_ablation import (
        filter_ablation_cases,
    )

    ablation = {"cases": [{"case_id": "case:one"}], "training_ready": False}
    packet = {
        "items": [
            {"review_id": "r:clue", "case_id": "case:one"},
            {"review_id": "r:control", "case_id": "case:one"},
        ]
    }
    decision = lambda review_id, relevance: {
        "review_id": review_id,
        "request_status": "completed",
        "media_alignment": "accept",
        "descriptor_grounding": "accept",
        "question_relevance": relevance,
    }
    audit = {
        "decisions": [decision("r:clue", "relevant"), decision("r:control", "relevant")]
    }
    key = {
        "items": [
            {"review_id": "r:clue", "gt_interval_role": "clue"},
            {"review_id": "r:control", "gt_interval_role": "control"},
        ]
    }
    filtered, report = filter_ablation_cases(ablation, packet, audit, key)
    assert filtered["cases"] == []
    assert (
        report["exclusion_reason_case_counts"]["control_contaminated_or_inconclusive"]
        == 1
    )
    assert filtered["training_ready"] is False
