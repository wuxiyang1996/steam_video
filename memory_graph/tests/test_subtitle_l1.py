from __future__ import annotations

from pathlib import Path

from memory_graph.subtitle_l1 import (
    SUBTITLE_ENRICHMENT_SCHEMA,
    enrich_video_l1_payload_with_subtitles,
    parse_srt,
)
from memory_graph.video_l1 import PayloadVideoL1Provider
from steam_video_new.implicit_world_model.reasoning_v2.evidence.legacy_adapter import (
    record_from_memory_node,
)


def _subtitle(path: Path) -> None:
    path.write_text(
        "1\n00:00:01,000 --> 00:00:02,500\nblueberry and grape\n\n"
        "2\n00:00:04,000 --> 00:00:05,000\nunrelated later cue\n",
        encoding="utf-8",
    )


def _payload() -> dict:
    return {
        "duration_s": 6.0,
        "nodes": [
            {
                "node_id": "node:a",
                "video_id": "video",
                "time_span": {"start_s": 1.2, "end_s": 2.0},
                "node_type": "observation",
                "text": "person holds ice cream",
                "embedding_ref": {
                    "path": "stale.npy",
                    "model": "stale",
                    "dimension": 2,
                    "dtype": "float32",
                    "normalized": True,
                    "row_index": 0,
                },
                "provenance": {"layer": "L1"},
                "metadata": {
                    "predicate": "person holds ice cream",
                    "action_kind": "contact",
                    "participants": [],
                    "states": [],
                },
            },
            {
                "node_id": "node:b",
                "video_id": "video",
                "time_span": {"start_s": 2.5, "end_s": 3.0},
                "node_type": "observation",
                "text": "person waits",
                "provenance": {"layer": "L1"},
                "metadata": {
                    "predicate": "person waits",
                    "semantic_key": "existing visual routing key",
                    "action_kind": "other",
                    "participants": [],
                    "states": [],
                },
            },
        ],
        "windowing": {"question_independent": True},
    }


def test_parse_and_enrich_subtitle_l1_is_time_aligned_and_question_independent(
    tmp_path,
) -> None:
    subtitle = tmp_path / "video.srt"
    _subtitle(subtitle)

    cues = parse_srt(subtitle)
    enriched = enrich_video_l1_payload_with_subtitles(_payload(), subtitle)

    assert len(cues) == 2
    assert enriched["descriptor_enrichment"]["schema_version"] == (
        SUBTITLE_ENRICHMENT_SCHEMA
    )
    assert enriched["descriptor_enrichment"]["aligned_node_count"] == 1
    first, second = enriched["nodes"]
    assert first["metadata"]["transcript"] == "blueberry and grape"
    assert "aligned subtitle: blueberry and grape" in first["metadata"][
        "semantic_key"
    ]
    assert "Time-aligned subtitle: blueberry and grape" in first["text"]
    assert first["embedding_ref"] is None
    assert first["provenance"]["subtitle_enrichment"]["question_independent"]
    assert "transcript" not in second["metadata"]
    assert second["metadata"]["semantic_key"] == "existing visual routing key"


def test_subtitle_address_is_bounded_but_grounded_value_is_read_only(tmp_path) -> None:
    subtitle = tmp_path / "video.srt"
    _subtitle(subtitle)
    enriched = enrich_video_l1_payload_with_subtitles(_payload(), subtitle)
    result = PayloadVideoL1Provider(enriched).extract(
        video_path="unused", video_id="video"
    )
    record = record_from_memory_node(result.nodes[0])

    assert len(record.address.semantic_key) <= 480
    assert "blueberry and grape" in record.address.semantic_key
    assert "Time-aligned subtitle: blueberry and grape" in record.value.descriptor
