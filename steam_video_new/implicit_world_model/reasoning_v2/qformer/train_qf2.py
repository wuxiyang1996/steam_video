"""Train QF2 on frozen QF1 slots with cross-video trusted negatives only."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .evaluation import permutation_consistency, retrieval_metrics
from .feature_store import sha256_file
from .losses import trusted_multi_positive_retrieval_loss
from .model import ConditionedProposalQFormer, NodeScoreHead
from .qf1_cache import QF1Cache
from .question_store import QuestionEmbeddingStore


class _RetrievalDataset(Dataset[dict[str, Any]]):
    def __init__(self, examples: Sequence[dict[str, Any]], cache: QF1Cache, questions: QuestionEmbeddingStore) -> None:
        self.examples = tuple(examples)
        self.cache = cache
        self.questions = questions

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        example = self.examples[index]
        return {
            "case_id": example["case_id"],
            "video_id": example["video_id"],
            "question": torch.from_numpy(
                np.array(self.questions.case(example["case_id"]), copy=True)
            ).float(),
            "nodes": torch.from_numpy(
                np.stack(
                    [np.array(self.cache.node(node_id), copy=True) for node_id in example["candidate_node_ids"]]
                )
            ).float(),
            "positive": torch.tensor(example["positive_mask"], dtype=torch.bool),
            "negative": torch.tensor(example["negative_mask"], dtype=torch.bool),
        }


def _collate(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    max_nodes = max(row["nodes"].shape[0] for row in rows)
    num_queries, hidden_size = rows[0]["nodes"].shape[1:]
    nodes = torch.zeros(len(rows), max_nodes, num_queries, hidden_size)
    positive = torch.zeros(len(rows), max_nodes, dtype=torch.bool)
    negative = torch.zeros_like(positive)
    for index, row in enumerate(rows):
        count = row["nodes"].shape[0]
        nodes[index, :count] = row["nodes"]
        positive[index, :count] = row["positive"]
        negative[index, :count] = row["negative"]
    return {
        "case_ids": tuple(row["case_id"] for row in rows),
        "video_ids": tuple(row["video_id"] for row in rows),
        "question": torch.stack([row["question"] for row in rows]),
        "nodes": nodes,
        "positive": positive,
        "negative": negative,
    }


def train_qf2(
    *,
    qf1_cache_manifest: Path,
    feature_manifest: Path,
    labels_path: Path,
    question_manifest: Path,
    validation_qf1_cache_manifest: Path | None = None,
    validation_feature_manifest: Path | None = None,
    validation_labels_path: Path | None = None,
    validation_question_manifest: Path | None = None,
    output_dir: Path,
    device: str,
    num_layers: int = 2,
    num_heads: int = 12,
    batch_size: int = 8,
    epochs: int = 10,
    learning_rate: float = 1e-4,
    trusted_negatives: int = 32,
    seed: int = 23,
    model_variant: str = "b3_frozen_qf1_eight_slots",
) -> dict[str, Any]:
    _seed(seed)
    labels_path = labels_path.expanduser().resolve()
    validation_labels_path = Path(validation_labels_path or labels_path).expanduser().resolve()
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    validation_labels = json.loads(validation_labels_path.read_text(encoding="utf-8"))
    if labels.get("answer_fields_present") is not False or validation_labels.get("answer_fields_present") is not False:
        raise ValueError("retrieval labels failed answer leakage gate")
    train_cache = QF1Cache(qf1_cache_manifest, feature_manifest_path=feature_manifest)
    validation_cache = QF1Cache(
        validation_qf1_cache_manifest or qf1_cache_manifest,
        feature_manifest_path=validation_feature_manifest or feature_manifest,
    )
    train_questions = QuestionEmbeddingStore(question_manifest, labels_path=labels_path)
    validation_questions = QuestionEmbeddingStore(
        validation_question_manifest or question_manifest,
        labels_path=validation_labels_path,
    )
    if train_questions.dimension != validation_questions.dimension:
        raise ValueError("train and validation question embedding dimensions differ")
    train_examples = _build_split_examples(
        labels.get("records") or (), train_cache, split="train", trusted_negatives=trusted_negatives, seed=seed
    )
    validation_examples = _build_split_examples(
        validation_labels.get("records") or (), validation_cache, split="validation", trusted_negatives=trusted_negatives, seed=seed
    )
    validation_ranking_examples = _build_ranking_examples(
        validation_labels.get("records") or (), validation_cache, split="validation"
    )
    train_loader = DataLoader(
        _RetrievalDataset(train_examples, train_cache, train_questions),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=_collate,
    )
    validation_loader = DataLoader(
        _RetrievalDataset(validation_examples, validation_cache, validation_questions),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_collate,
    )
    validation_ranking_loader = DataLoader(
        _RetrievalDataset(validation_ranking_examples, validation_cache, validation_questions),
        batch_size=1,
        shuffle=False,
        collate_fn=_collate,
    )
    target_device = torch.device(device)
    hidden_size = train_cache.manifest.hidden_size
    if validation_cache.manifest.hidden_size != hidden_size:
        raise ValueError("train and validation QF1 hidden sizes differ")
    question_projector = nn.Sequential(
        nn.LayerNorm(train_questions.dimension), nn.Linear(train_questions.dimension, hidden_size)
    ).to(target_device)
    qf2 = ConditionedProposalQFormer(
        hidden_size=hidden_size,
        num_queries=8,
        num_layers=num_layers,
        num_heads=num_heads,
    ).to(target_device)
    scorer = NodeScoreHead(hidden_size).to(target_device)
    modules = (question_projector, qf2, scorer)
    optimizer = torch.optim.AdamW(
        [parameter for module in modules for parameter in module.parameters()],
        lr=learning_rate,
        weight_decay=0.05,
    )
    initial = _evaluate(modules, validation_loader, target_device)
    initial_ranking = _evaluate_ranking(modules, validation_ranking_loader, target_device)
    history: list[dict[str, Any]] = []
    best_loss = float("inf")
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "qf2_best.pt"
    for epoch in range(1, epochs + 1):
        for module in modules:
            module.train()
        total = 0.0
        count = 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            scores, positive, negative = _forward(modules, batch, target_device)
            loss = trusted_multi_positive_retrieval_loss(scores, positive, negative)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [parameter for module in modules for parameter in module.parameters()], 1.0
            )
            optimizer.step()
            positive_count = int(positive.sum())
            total += float(loss.detach()) * positive_count
            count += positive_count
        validation = _evaluate(modules, validation_loader, target_device)
        history.append(
            {"epoch": epoch, "train_loss": total / max(count, 1), "validation": validation}
        )
        if validation["loss"] < best_loss:
            best_loss = validation["loss"]
            torch.save(
                {
                    "schema_version": "steam-qformer-qf2-checkpoint/v0.1",
                    "question_projector": question_projector.state_dict(),
                    "qf2": qf2.state_dict(),
                    "scorer": scorer.state_dict(),
                    "config": {
                        "model_variant": model_variant,
                        "question_dimension": train_questions.dimension,
                        "hidden_size": hidden_size,
                        "num_queries": 8,
                        "memory_tokens": train_cache.manifest.num_queries,
                        "num_layers": num_layers,
                        "num_heads": num_heads,
                    },
                    "qf1_cache_checksum": sha256_file(Path(qf1_cache_manifest)),
                    "validation": validation,
                },
                checkpoint_path,
            )
    train_videos = {row["video_id"] for row in train_examples}
    validation_videos = {row["video_id"] for row in validation_examples}
    final_validation = min(history, key=lambda row: row["validation"]["loss"])["validation"]
    best_checkpoint = torch.load(checkpoint_path, map_location=target_device, weights_only=True)
    question_projector.load_state_dict(best_checkpoint["question_projector"])
    qf2.load_state_dict(best_checkpoint["qf2"])
    scorer.load_state_dict(best_checkpoint["scorer"])
    heldout_ranking = _evaluate_ranking(modules, validation_ranking_loader, target_device)
    report = {
        "schema_version": "steam-qformer-qf2-training/v0.1",
        "model_variant": model_variant,
        "checkpoint": str(checkpoint_path),
        "qf1_cache_manifest": str(Path(qf1_cache_manifest).resolve()),
        "labels": str(labels_path),
        "validation_qf1_cache_manifest": str(Path(validation_qf1_cache_manifest or qf1_cache_manifest).resolve()),
        "validation_labels": str(validation_labels_path),
        "config": {
            "num_queries": 8,
            "memory_tokens": train_cache.manifest.num_queries,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "batch_size": batch_size,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "trusted_negatives": trusted_negatives,
            "seed": seed,
        },
        "data": {
            "train_questions": len(train_examples),
            "validation_questions": len(validation_examples),
            "train_videos": len(train_videos),
            "validation_videos": len(validation_videos),
        },
        "initial_validation": initial,
        "initial_heldout_full_candidate_ranking": initial_ranking,
        "history": history,
        "best_validation": final_validation,
        "heldout_full_candidate_ranking": heldout_ranking,
        "gates": {
            "video_disjoint": train_videos.isdisjoint(validation_videos),
            "finite_losses": bool(np.isfinite([initial["loss"], best_loss]).all()),
            "validation_improved": best_loss < initial["loss"],
            "permutation_max_error_lte_1e-5": heldout_ranking["permutation"]["max_absolute_error"] <= 1e-5,
            "answer_leakage_absent": labels.get("answer_fields_present") is False and validation_labels.get("answer_fields_present") is False,
        },
    }
    report["passed"] = all(report["gates"].values())
    (output_dir / "training_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def _build_split_examples(records, cache: QF1Cache, *, split: str, trusted_negatives: int, seed: int):
    rows_by_video: dict[str, list[str]] = {}
    for row in cache.manifest.rows:
        rows_by_video.setdefault(row.video_id, []).append(row.node_id)
    split_videos: dict[str, set[str]] = {}
    for record in records:
        split_videos.setdefault(str(record.get("split") or ""), set()).add(str(record["video_id"]))
    outputs: list[dict[str, Any]] = []
    for record in records:
        record_split = str(record.get("split") or "")
        if record_split != split:
            continue
        video_id = str(record["video_id"])
        positives = [node for node in record["positive_node_ids"] if node in cache]
        negative_pool = [
            node
            for other_video in sorted(split_videos.get(record_split, ()))
            if other_video != video_id
            for node in rows_by_video.get(other_video, ())
        ]
        if not positives or not negative_pool:
            continue
        case_id = str(record["case_id"])
        digest = hashlib.sha256(f"{seed}:{case_id}".encode()).digest()
        rng = random.Random(int.from_bytes(digest[:8], "big"))
        negatives = rng.sample(negative_pool, min(trusted_negatives, len(negative_pool)))
        candidates = positives + negatives
        labels = [True] * len(positives) + [False] * len(negatives)
        order = list(range(len(candidates)))
        rng.shuffle(order)
        outputs.append(
            {
                "case_id": case_id,
                "video_id": video_id,
                "candidate_node_ids": [candidates[index] for index in order],
                "positive_mask": [labels[index] for index in order],
                "negative_mask": [not labels[index] for index in order],
            }
        )
    if not outputs:
        raise ValueError(f"QF2 requires non-empty {split} records with cross-video negatives")
    return outputs


def _build_ranking_examples(records, cache: QF1Cache, *, split: str):
    outputs: list[dict[str, Any]] = []
    for record in records:
        if str(record.get("split") or "") != split:
            continue
        positives = set(node for node in record["positive_node_ids"] if node in cache)
        candidates = [node for node in record["candidate_node_ids"] if node in cache]
        if not positives or not candidates:
            continue
        positive_mask = [node in positives for node in candidates]
        outputs.append(
            {
                "case_id": str(record["case_id"]),
                "video_id": str(record["video_id"]),
                "candidate_node_ids": candidates,
                "positive_mask": positive_mask,
                # This mask means valid-for-ranking here, not a trusted label.
                "negative_mask": [not value for value in positive_mask],
            }
        )
    if not outputs:
        raise ValueError(f"QF2 requires non-empty {split} full-candidate ranking records")
    return outputs


def _forward(modules, batch, device):
    question_projector, qf2, scorer = modules
    nodes = batch["nodes"].to(device)
    question = question_projector(batch["question"].to(device)).unsqueeze(1)
    batch_size, node_count, num_queries, hidden_size = nodes.shape
    flat_nodes = nodes.reshape(batch_size * node_count, num_queries, hidden_size)
    repeated_question = question[:, None].expand(-1, node_count, -1, -1).reshape(
        batch_size * node_count, 1, hidden_size
    )
    proposals = qf2(flat_nodes, repeated_question).reshape(
        batch_size, node_count, qf2.num_queries, hidden_size
    )
    scores = scorer(proposals, question)
    return scores, batch["positive"].to(device), batch["negative"].to(device)


@torch.no_grad()
def _evaluate(modules, loader, device):
    for module in modules:
        module.eval()
    score_rows: list[np.ndarray] = []
    positive_rows: list[np.ndarray] = []
    valid_rows: list[np.ndarray] = []
    total = 0.0
    count = 0
    permutation = {"max_absolute_error": 0.0, "mean_absolute_error": 0.0}
    for batch_index, batch in enumerate(loader):
        scores, positive, negative = _forward(modules, batch, device)
        loss = trusted_multi_positive_retrieval_loss(scores, positive, negative)
        positive_count = int(positive.sum())
        total += float(loss) * positive_count
        count += positive_count
        valid = positive | negative
        for row_score, row_positive, row_valid in zip(scores.cpu().numpy(), positive.cpu().numpy(), valid.cpu().numpy()):
            keep = int(row_valid.sum())
            score_rows.append(row_score[:keep])
            positive_rows.append(row_positive[:keep])
            valid_rows.append(row_valid[:keep])
        if batch_index == 0:
            order = torch.arange(scores.shape[1] - 1, -1, -1, device=device)
            permuted_batch = dict(batch)
            permuted_batch["nodes"] = batch["nodes"][:, order.cpu()]
            permuted_scores, _, _ = _forward(modules, permuted_batch, device)
            expected_permutation = np.tile(order.cpu().numpy(), (scores.shape[0], 1))
            permutation = permutation_consistency(
                scores.cpu().numpy(), permuted_scores.cpu().numpy(), expected_permutation
            )
    width = max(len(row) for row in score_rows)
    score_matrix = np.full((len(score_rows), width), -np.inf, dtype=np.float32)
    positive_matrix = np.zeros((len(score_rows), width), dtype=np.bool_)
    valid_matrix = np.zeros_like(positive_matrix)
    for index, (scores, positives, valid) in enumerate(zip(score_rows, positive_rows, valid_rows)):
        score_matrix[index, : len(scores)] = scores
        positive_matrix[index, : len(positives)] = positives
        valid_matrix[index, : len(valid)] = valid
    return {
        "loss": total / max(count, 1),
        "metrics": retrieval_metrics(score_matrix, positive_matrix, valid_matrix),
        "permutation": permutation,
    }


@torch.no_grad()
def _evaluate_ranking(modules, loader, device):
    for module in modules:
        module.eval()
    score_rows: list[np.ndarray] = []
    positive_rows: list[np.ndarray] = []
    valid_rows: list[np.ndarray] = []
    permutation_errors: list[dict[str, float]] = []
    for batch in loader:
        scores, positive, ranking_nonpositive = _forward(modules, batch, device)
        valid = positive | ranking_nonpositive
        score_rows.append(scores[0].cpu().numpy())
        positive_rows.append(positive[0].cpu().numpy())
        valid_rows.append(valid[0].cpu().numpy())
        order = torch.arange(scores.shape[1] - 1, -1, -1)
        permuted_batch = dict(batch)
        permuted_batch["nodes"] = batch["nodes"][:, order]
        permuted_scores, _, _ = _forward(modules, permuted_batch, device)
        permutation_errors.append(
            permutation_consistency(
                scores.cpu().numpy(),
                permuted_scores.cpu().numpy(),
                np.tile(order.numpy(), (scores.shape[0], 1)),
            )
        )
    width = max(len(row) for row in score_rows)
    score_matrix = np.full((len(score_rows), width), -np.inf, dtype=np.float32)
    positive_matrix = np.zeros((len(score_rows), width), dtype=np.bool_)
    valid_matrix = np.zeros_like(positive_matrix)
    for index, (scores, positives, valid) in enumerate(zip(score_rows, positive_rows, valid_rows)):
        score_matrix[index, : len(scores)] = scores
        positive_matrix[index, : len(positives)] = positives
        valid_matrix[index, : len(valid)] = valid
    return {
        "candidate_semantics": "all_same_video_nodes; nonpositives_unlabeled_ranking_only",
        "question_count": len(score_rows),
        "metrics": retrieval_metrics(score_matrix, positive_matrix, valid_matrix),
        "permutation": {
            "max_absolute_error": max(row["max_absolute_error"] for row in permutation_errors),
            "mean_absolute_error": float(np.mean([row["mean_absolute_error"] for row in permutation_errors])),
        },
    }


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qf1-cache-manifest", required=True, type=Path)
    parser.add_argument("--feature-manifest", required=True, type=Path)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--question-manifest", required=True, type=Path)
    parser.add_argument("--validation-qf1-cache-manifest", type=Path)
    parser.add_argument("--validation-feature-manifest", type=Path)
    parser.add_argument("--validation-labels", type=Path)
    parser.add_argument("--validation-question-manifest", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--trusted-negatives", type=int, default=32)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--model-variant", default="b3_frozen_qf1_eight_slots")
    args = parser.parse_args(argv)
    report = train_qf2(
        qf1_cache_manifest=args.qf1_cache_manifest,
        feature_manifest=args.feature_manifest,
        labels_path=args.labels,
        question_manifest=args.question_manifest,
        validation_qf1_cache_manifest=args.validation_qf1_cache_manifest,
        validation_feature_manifest=args.validation_feature_manifest,
        validation_labels_path=args.validation_labels,
        validation_question_manifest=args.validation_question_manifest,
        output_dir=args.output_dir,
        device=args.device,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        trusted_negatives=args.trusted_negatives,
        seed=args.seed,
        model_variant=args.model_variant,
    )
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
