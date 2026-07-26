"""Build a split-safe corpus of grounded executed-transition descriptors."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


SCHEMA = "steam-grounded-action-transition-corpus/v0.1"
DESCRIPTOR_FIELDS = (
    "summary",
    "visible_entities",
    "visible_actions",
    "visible_states",
    "readable_text",
    "evidence_frames",
    "frame_index_status",
)


def build_grounded_transition_corpus(dataset: dict[str, Any]) -> dict[str, Any]:
    """Extract real observations and categorical deltas without hidden GT joins."""

    records: list[dict[str, Any]] = []
    unavailable: list[dict[str, str]] = []
    seen: set[str] = set()
    videos_by_split: dict[str, set[str]] = {}
    for case in dataset.get("cases") or []:
        case_id = str(case["case_id"])
        video_id = str(case["video_id"])
        split = str(case["split"])
        videos_by_split.setdefault(split, set()).add(video_id)
        question = str((case.get("planner_input") or {}).get("question") or "")
        if not question:
            raise ValueError(f"case {case_id} has no question")
        for transition in case.get("executed_transitions") or []:
            transition_id = str(transition["transition_id"])
            if transition_id in seen:
                raise ValueError(f"duplicate transition ID: {transition_id}")
            seen.add(transition_id)
            observation = transition.get("real_observation") or {}
            status = str(observation.get("descriptor_status") or "")
            descriptor = observation.get("descriptor")
            if status != "grounded_qwen_vl_read" or not isinstance(descriptor, dict):
                unavailable.append(
                    {
                        "transition_id": transition_id,
                        "case_id": case_id,
                        "video_id": video_id,
                        "split": split,
                        "descriptor_status": status or "missing",
                    }
                )
                continue
            action = transition.get("action") or {}
            target = transition.get("target") or {}
            delta = target.get("belief_delta") or {}
            embedding_ref = (target.get("observation_descriptor") or {}).get(
                "embedding_ref"
            )
            if not isinstance(embedding_ref, dict):
                raise ValueError(f"grounded transition {transition_id} lacks embedding")
            grounded_descriptor = {
                field: descriptor[field]
                for field in DESCRIPTOR_FIELDS
                if field in descriptor
            }
            if not str(grounded_descriptor.get("summary") or "").strip():
                raise ValueError(f"grounded transition {transition_id} has no summary")
            records.append(
                {
                    "record_id": f"gtcache:{_digest(transition_id)}",
                    "source_transition_id": transition_id,
                    "case_id": case_id,
                    "video_id": video_id,
                    "split": split,
                    "training_context": {
                        "question": question,
                        "checkpoint": transition.get("checkpoint"),
                        "action": action,
                    },
                    "real_transition_target": {
                        "observation_descriptor": grounded_descriptor,
                        "observed_modalities": observation.get("observed_modalities") or [],
                        "producer": observation.get("producer"),
                        "embedding_ref": embedding_ref,
                        "categorical_belief_delta": delta,
                    },
                    "target_is_real_not_imagined": True,
                    "hidden_clue_interval_or_answer_present": False,
                }
            )
    splits = sorted(videos_by_split)
    overlap = {
        f"{left}:{right}": sorted(videos_by_split[left] & videos_by_split[right])
        for index, left in enumerate(splits)
        for right in splits[index + 1 :]
        if videos_by_split[left] & videos_by_split[right]
    }
    if overlap:
        raise ValueError(f"video-disjoint split violation: {overlap}")
    split_counts = {
        split: {
            "record_count": sum(row["split"] == split for row in records),
            "video_count": len(videos_by_split[split]),
        }
        for split in splits
    }
    return {
        "schema_version": SCHEMA,
        "dataset_id": dataset.get("dataset_id"),
        "source_dataset_sha256": _checksum(dataset),
        "purpose": "future IWM post-training or audited descriptor retrieval",
        "allowed_consumers": ["iwm_training", "offline_iwm_evaluation"],
        "forbidden_consumers": [
            "l1_writer",
            "l1.5_graph_builder",
            "same_case_runtime_lookup",
            "hidden_evaluator",
        ],
        "split_contract": "video_disjoint",
        "model_generated_numeric_reward_present": False,
        "preference_target": "categorical_only",
        "training_performed": False,
        "record_count": len(records),
        "unavailable_count": len(unavailable),
        "split_counts": split_counts,
        "records": sorted(records, key=lambda row: row["source_transition_id"]),
        "unavailable": sorted(
            unavailable, key=lambda row: row["transition_id"]
        ),
    }


def _checksum(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:20]


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    result = build_grounded_transition_corpus(dataset)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "record_count": result["record_count"],
                "unavailable_count": result["unavailable_count"],
                "split_counts": result["split_counts"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
