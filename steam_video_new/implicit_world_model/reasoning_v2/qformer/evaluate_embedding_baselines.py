"""Evaluate frozen B0/B1 cosine baselines on full same-video candidate pools."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .evaluation import retrieval_metrics
from .feature_store import FourSlotFeatureStore
from .question_store import QuestionEmbeddingStore


def evaluate_embedding_baselines(
    *,
    feature_manifest: Path,
    labels_path: Path,
    question_manifest: Path,
    split: str = "validation",
) -> dict[str, Any]:
    labels_path = labels_path.expanduser().resolve()
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    if labels.get("answer_fields_present") is not False:
        raise ValueError("retrieval labels failed answer leakage gate")
    features = FourSlotFeatureStore(feature_manifest)
    questions = QuestionEmbeddingStore(question_manifest, labels_path=labels_path)
    if any(features.manifest.matrices[name].dimension != questions.dimension for name in ("caption", "entity_state", "visual")):
        raise ValueError("B0/B1 require caption/entity/visual and question embeddings in one frozen space")
    rows: list[dict[str, Any]] = []
    for record in labels.get("records") or ():
        if str(record.get("split") or "") != split:
            continue
        candidates = [node for node in record["candidate_node_ids"] if node in features]
        positives = set(record["positive_node_ids"])
        if not candidates or not any(node in positives for node in candidates):
            continue
        question = _normalize(np.array(questions.case(str(record["case_id"])), copy=True))
        caption_scores: list[float] = []
        mean_scores: list[float] = []
        for node_id in candidates:
            row = features.node(node_id)
            caption_scores.append(float(_normalize(row["features"]["caption"]) @ question))
            shared = [
                row["features"][name]
                for index, name in enumerate(("caption", "entity_state", "visual"))
                if bool(row["validity"][index])
            ]
            mean_scores.append(float(_normalize(np.mean(shared, axis=0)) @ question))
        rows.append(
            {
                "caption": np.asarray(caption_scores, dtype=np.float32),
                "mean": np.asarray(mean_scores, dtype=np.float32),
                "positive": np.asarray([node in positives for node in candidates], dtype=np.bool_),
            }
        )
    if not rows:
        raise ValueError(f"no {split} records available for embedding baselines")
    width = max(len(row["caption"]) for row in rows)
    positive = np.zeros((len(rows), width), dtype=np.bool_)
    valid = np.zeros_like(positive)
    scores = {
        "b0_caption_question_cosine": np.full((len(rows), width), -np.inf, dtype=np.float32),
        "b1_three_shared_space_slots_mean_question_cosine": np.full((len(rows), width), -np.inf, dtype=np.float32),
    }
    for index, row in enumerate(rows):
        count = len(row["caption"])
        positive[index, :count] = row["positive"]
        valid[index, :count] = True
        scores["b0_caption_question_cosine"][index, :count] = row["caption"]
        scores["b1_three_shared_space_slots_mean_question_cosine"][index, :count] = row["mean"]
    return {
        "schema_version": "steam-qformer-embedding-baselines/v0.1",
        "split": split,
        "question_count": len(rows),
        "candidate_semantics": "all_same_video_nodes; nonpositives_unlabeled_ranking_only",
        "b1_time_slot_excluded_reason": "time is 4D and not in the frozen Qwen semantic embedding space",
        "metrics": {
            name: retrieval_metrics(matrix, positive, valid) for name, matrix in scores.items()
        },
    }


def _normalize(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    return value / max(float(np.linalg.norm(value)), 1e-12)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-manifest", required=True, type=Path)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--question-manifest", required=True, type=Path)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    report = evaluate_embedding_baselines(
        feature_manifest=args.feature_manifest,
        labels_path=args.labels,
        question_manifest=args.question_manifest,
        split=args.split,
    )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
