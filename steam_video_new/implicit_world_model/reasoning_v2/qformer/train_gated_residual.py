"""Train A0-A4 visual-anchor four-feature gated residual baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .evaluation import clue_group_metrics, permutation_consistency, retrieval_metrics
from .feature_store import FourSlotFeatureStore, sha256_file
from .losses import trusted_group_retrieval_loss
from .model import (
    ConditionedProposalQFormer,
    GatedFourFeatureResidualScorer,
    NodeScoreHead,
)
from .qf1_cache import QF1Cache
from .question_store import QuestionEmbeddingStore
from .train_qf2 import _build_ranking_examples, _build_split_examples, _seed


VARIANTS = {
    "a0_visual": (False, False, False, False),
    "a1_visual_caption": (True, False, False, False),
    "a2_visual_caption_entity": (True, True, False, False),
    "a3_all_four_no_qf": (True, True, True, False),
    "a4_all_four_qf_residual": (True, True, True, True),
    # The residual branch reads four complete typed projector tokens.  It does
    # not use the caption/entity/time scalar-cosine residuals.
    "a5_visual_four_token_qf2": (False, False, False, True),
}


class _GatedDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        examples: Sequence[dict[str, Any]],
        features: FourSlotFeatureStore,
        questions: QuestionEmbeddingStore,
        qf_cache: QF1Cache | None,
    ) -> None:
        self.examples = tuple(examples)
        self.features = features
        self.questions = questions
        self.qf_cache = qf_cache

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        example = self.examples[index]
        question = np.array(self.questions.case(example["case_id"]), copy=True)
        question /= max(float(np.linalg.norm(question)), 1e-12)
        semantic: list[list[float]] = []
        time_values: list[np.ndarray] = []
        for node_id in example["candidate_node_ids"]:
            row = self.features.node(node_id)
            node_scores: list[float] = []
            for slot_index, name in enumerate(("caption", "entity_state", "visual")):
                if not bool(row["validity"][slot_index]):
                    node_scores.append(0.0)
                    continue
                value = np.asarray(row["features"][name], dtype=np.float32)
                value = value / max(float(np.linalg.norm(value)), 1e-12)
                node_scores.append(float(value @ question))
            semantic.append(node_scores)
            time_values.append(np.array(row["features"]["time"], copy=True))
        result = {
            "case_id": example["case_id"],
            "video_id": example["video_id"],
            "question": torch.from_numpy(question).float(),
            "semantic": torch.tensor(semantic, dtype=torch.float32),
            "time": torch.from_numpy(np.stack(time_values)).float(),
            "positive": torch.tensor(example["positive_mask"], dtype=torch.bool),
            "negative": torch.tensor(example["negative_mask"], dtype=torch.bool),
            "groups": torch.tensor(example["positive_group_masks"], dtype=torch.bool),
        }
        if self.qf_cache is not None:
            result["qf_nodes"] = torch.from_numpy(
                np.stack(
                    [
                        np.array(self.qf_cache.node(node_id), copy=True)
                        for node_id in example["candidate_node_ids"]
                    ]
                )
            ).float()
        return result


def _collate(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    max_nodes = max(row["semantic"].shape[0] for row in rows)
    max_groups = max(row["groups"].shape[0] for row in rows)
    semantic = torch.zeros(len(rows), max_nodes, 3)
    time_values = torch.zeros(len(rows), max_nodes, 4)
    positive = torch.zeros(len(rows), max_nodes, dtype=torch.bool)
    negative = torch.zeros_like(positive)
    groups = torch.zeros(len(rows), max_groups, max_nodes, dtype=torch.bool)
    group_validity = torch.zeros(len(rows), max_groups, dtype=torch.bool)
    qf_nodes = None
    if "qf_nodes" in rows[0]:
        query_count, hidden_size = rows[0]["qf_nodes"].shape[1:]
        qf_nodes = torch.zeros(len(rows), max_nodes, query_count, hidden_size)
    for index, row in enumerate(rows):
        node_count = row["semantic"].shape[0]
        group_count = row["groups"].shape[0]
        semantic[index, :node_count] = row["semantic"]
        time_values[index, :node_count] = row["time"]
        positive[index, :node_count] = row["positive"]
        negative[index, :node_count] = row["negative"]
        groups[index, :group_count, :node_count] = row["groups"]
        group_validity[index, :group_count] = True
        if qf_nodes is not None:
            qf_nodes[index, :node_count] = row["qf_nodes"]
    result = {
        "case_ids": tuple(row["case_id"] for row in rows),
        "video_ids": tuple(row["video_id"] for row in rows),
        "question": torch.stack([row["question"] for row in rows]),
        "semantic": semantic,
        "time": time_values,
        "positive": positive,
        "negative": negative,
        "groups": groups,
        "group_validity": group_validity,
    }
    if qf_nodes is not None:
        result["qf_nodes"] = qf_nodes
    return result


def train_gated_residual(
    *,
    variant: str,
    train_feature_manifest: Path,
    validation_feature_manifest: Path,
    train_labels_path: Path,
    validation_labels_path: Path,
    train_question_manifest: Path,
    validation_question_manifest: Path,
    output_dir: Path,
    device: str,
    train_qf_cache_manifest: Path | None = None,
    validation_qf_cache_manifest: Path | None = None,
    num_layers: int = 1,
    num_heads: int = 12,
    batch_size: int = 8,
    epochs: int = 20,
    learning_rate: float = 3e-4,
    trusted_negatives: int = 32,
    seed: int = 31,
    selection_fraction: float = 0.2,
    selection_seed: int = 1729,
    same_video_negatives: int = 0,
    same_video_negative_mining: str = "random",
    qf2_init_checkpoint: Path | None = None,
) -> dict[str, Any]:
    if variant not in VARIANTS:
        raise ValueError(f"unknown gated baseline variant: {variant}")
    if same_video_negative_mining not in {"random", "visual_hard"}:
        raise ValueError("same-video negative mining must be random or visual_hard")
    _seed(seed)
    enable_caption, enable_entity, enable_time, enable_qf = VARIANTS[variant]
    train_labels_path = train_labels_path.expanduser().resolve()
    validation_labels_path = validation_labels_path.expanduser().resolve()
    train_labels = json.loads(train_labels_path.read_text(encoding="utf-8"))
    validation_labels = json.loads(validation_labels_path.read_text(encoding="utf-8"))
    for payload in (train_labels, validation_labels):
        if payload.get("schema_version") not in {
            "steam-qformer-retrieval-labels/v0.2",
            "steam-qformer-retrieval-labels/v0.3",
        }:
            raise ValueError("gated residual training requires clue-group labels")
        if payload.get("answer_fields_present") is not False:
            raise ValueError("retrieval labels failed answer leakage gate")
    train_features = FourSlotFeatureStore(train_feature_manifest)
    validation_features = FourSlotFeatureStore(validation_feature_manifest)
    train_questions = QuestionEmbeddingStore(
        train_question_manifest, labels_path=train_labels_path
    )
    validation_questions = QuestionEmbeddingStore(
        validation_question_manifest, labels_path=validation_labels_path
    )
    if train_questions.dimension != validation_questions.dimension:
        raise ValueError("train and validation question dimensions differ")
    train_qf = validation_qf = None
    if enable_qf:
        if train_qf_cache_manifest is None or validation_qf_cache_manifest is None:
            raise ValueError("A4 requires train and validation QF1 caches")
        train_qf = QF1Cache(
            train_qf_cache_manifest, feature_manifest_path=train_feature_manifest
        )
        validation_qf = QF1Cache(
            validation_qf_cache_manifest,
            feature_manifest_path=validation_feature_manifest,
        )
    fit_records, selection_records = _partition_train_records(
        train_labels["records"],
        fraction=selection_fraction,
        seed=selection_seed,
    )
    train_examples = _examples_with_groups(
        fit_records,
        train_qf or train_features,
        split="fit",
        trusted_negatives=trusted_negatives,
        seed=seed,
        ranking=False,
        same_video_negatives=same_video_negatives,
        same_video_negative_mining=same_video_negative_mining,
        feature_store=train_features,
        question_store=train_questions,
    )
    selection_examples = _examples_with_groups(
        selection_records,
        train_qf or train_features,
        split="selection",
        trusted_negatives=trusted_negatives,
        seed=seed,
        ranking=False,
        same_video_negatives=same_video_negatives,
        same_video_negative_mining=same_video_negative_mining,
        feature_store=train_features,
        question_store=train_questions,
    )
    selection_ranking_examples = _examples_with_groups(
        selection_records,
        train_qf or train_features,
        split="selection",
        trusted_negatives=trusted_negatives,
        seed=seed,
        ranking=True,
        same_video_negatives=0,
    )
    validation_examples = _examples_with_groups(
        validation_labels["records"],
        validation_qf or validation_features,
        split="validation",
        trusted_negatives=trusted_negatives,
        seed=seed,
        ranking=False,
        same_video_negatives=same_video_negatives,
        same_video_negative_mining=same_video_negative_mining,
        feature_store=validation_features,
        question_store=validation_questions,
    )
    ranking_examples = _examples_with_groups(
        validation_labels["records"],
        validation_qf or validation_features,
        split="validation",
        trusted_negatives=trusted_negatives,
        seed=seed,
        ranking=True,
        same_video_negatives=0,
    )
    train_loader = DataLoader(
        _GatedDataset(train_examples, train_features, train_questions, train_qf),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=_collate,
    )
    validation_loader = DataLoader(
        _GatedDataset(
            validation_examples, validation_features, validation_questions, validation_qf
        ),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_collate,
    )
    selection_loader = DataLoader(
        _GatedDataset(selection_examples, train_features, train_questions, train_qf),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_collate,
    )
    selection_ranking_loader = DataLoader(
        _GatedDataset(
            selection_ranking_examples, train_features, train_questions, train_qf
        ),
        batch_size=1,
        shuffle=False,
        collate_fn=_collate,
    )
    ranking_loader = DataLoader(
        _GatedDataset(
            ranking_examples, validation_features, validation_questions, validation_qf
        ),
        batch_size=1,
        shuffle=False,
        collate_fn=_collate,
    )
    target_device = torch.device(device)
    residual = GatedFourFeatureResidualScorer(
        train_questions.dimension,
        enable_caption=enable_caption,
        enable_entity=enable_entity,
        enable_time=enable_time,
        enable_qf=enable_qf,
    ).to(target_device)
    modules: dict[str, nn.Module] = {"residual": residual}
    if enable_qf:
        assert train_qf is not None
        hidden_size = train_qf.manifest.hidden_size
        modules.update(
            {
                "qf_question": nn.Sequential(
                    nn.LayerNorm(train_questions.dimension),
                    nn.Linear(train_questions.dimension, hidden_size),
                ).to(target_device),
                "qf2": ConditionedProposalQFormer(
                    hidden_size=hidden_size,
                    num_queries=8,
                    num_layers=num_layers,
                    num_heads=num_heads,
                ).to(target_device),
                "qf_scorer": NodeScoreHead(hidden_size).to(target_device),
            }
        )
        if qf2_init_checkpoint is not None:
            _initialize_qf2_from_qf1(
                modules,
                qf2_init_checkpoint,
                expected_layers=num_layers,
                expected_hidden_size=hidden_size,
            )
    initial_selection_trusted = _evaluate_trusted(
        modules, selection_loader, target_device
    )
    initial_selection_ranking = _evaluate_ranking(
        modules, selection_ranking_loader, target_device
    )
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "best.pt"
    best_key = _selection_key(initial_selection_ranking)
    best_epoch = 0
    _save_checkpoint(
        checkpoint_path, modules, variant, 0, initial_selection_ranking
    )
    history: list[dict[str, Any]] = []
    if variant != "a0_visual":
        optimizer = torch.optim.AdamW(
            [parameter for module in modules.values() for parameter in module.parameters()],
            lr=learning_rate,
            weight_decay=0.01,
        )
        for epoch in range(1, epochs + 1):
            for module in modules.values():
                module.train()
            total = 0.0
            group_count = 0
            for batch in train_loader:
                optimizer.zero_grad(set_to_none=True)
                scores, positive_groups, group_validity, negatives, _ = _forward(
                    modules, batch, target_device
                )
                loss = trusted_group_retrieval_loss(
                    scores, positive_groups, group_validity, negatives
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [
                        parameter
                        for module in modules.values()
                        for parameter in module.parameters()
                    ],
                    1.0,
                )
                optimizer.step()
                count = int(group_validity.sum())
                total += float(loss.detach()) * count
                group_count += count
            trusted = _evaluate_trusted(modules, selection_loader, target_device)
            ranking = _evaluate_ranking(
                modules, selection_ranking_loader, target_device
            )
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": total / max(group_count, 1),
                    "trusted_validation": trusted,
                    "full_candidate_ranking": ranking,
                }
            )
            key = _selection_key(ranking)
            if key > best_key:
                best_key = key
                best_epoch = epoch
                _save_checkpoint(checkpoint_path, modules, variant, epoch, ranking)
    checkpoint = torch.load(checkpoint_path, map_location=target_device, weights_only=True)
    for name, module in modules.items():
        module.load_state_dict(checkpoint["modules"][name])
    best_selection_ranking = _evaluate_ranking(
        modules, selection_ranking_loader, target_device
    )
    heldout_trusted = _evaluate_trusted(modules, validation_loader, target_device)
    heldout_ranking = _evaluate_ranking(
        modules, ranking_loader, target_device, include_per_case=True
    )
    train_videos = {row["video_id"] for row in train_examples}
    selection_videos = {row["video_id"] for row in selection_examples}
    validation_videos = {row["video_id"] for row in validation_examples}
    report = {
        "schema_version": "steam-qformer-gated-residual/v0.2",
        "variant": variant,
        "checkpoint": str(checkpoint_path),
        "config": {
            "features": {
                "visual_anchor": True,
                "caption_residual": enable_caption,
                "entity_residual": enable_entity,
                "time_residual": enable_time,
                "qf_residual": enable_qf,
            },
            "qf_memory_tokens": train_qf.manifest.num_queries if train_qf else 0,
            "num_layers": num_layers if enable_qf else 0,
            "batch_size": batch_size,
            "epochs": 0 if variant == "a0_visual" else epochs,
            "learning_rate": learning_rate,
            "trusted_negatives": trusted_negatives,
            "seed": seed,
            "selection_fraction": selection_fraction,
            "selection_seed": selection_seed,
            "same_video_negatives": same_video_negatives,
            "same_video_negative_mining": same_video_negative_mining,
            "qf2_initialization": (
                "qf1_and_caption_projector"
                if qf2_init_checkpoint is not None
                else "random"
            ),
        },
        "data": {
            "train_questions": len(train_examples),
            "selection_questions": len(selection_examples),
            "validation_questions": len(validation_examples),
            "train_videos": len(train_videos),
            "selection_videos": len(selection_videos),
            "validation_videos": len(validation_videos),
        },
        "initial_selection_trusted": initial_selection_trusted,
        "initial_selection_ranking": initial_selection_ranking,
        "history": history,
        "best_epoch": best_epoch,
        "best_selection_ranking": best_selection_ranking,
        "heldout_trusted": heldout_trusted,
        "heldout_ranking": heldout_ranking,
        "gates": {
            "fit_selection_video_disjoint": train_videos.isdisjoint(selection_videos),
            "train_heldout_video_disjoint": (train_videos | selection_videos).isdisjoint(
                validation_videos
            ),
            "finite": bool(np.isfinite(heldout_ranking["node_union_metrics"]["mrr"])),
            "permutation": heldout_ranking["permutation"]["max_absolute_error"] <= 1e-5,
            "answer_leakage_absent": True,
            "not_worse_than_visual_anchor_on_selection": _selection_key(
                best_selection_ranking
            )
            >= _selection_key(initial_selection_ranking),
        },
    }
    report["passed"] = all(report["gates"].values())
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def _examples_with_groups(
    records: Sequence[Mapping[str, Any]],
    node_source: Any,
    *,
    split: str,
    trusted_negatives: int,
    seed: int,
    ranking: bool,
    same_video_negatives: int = 0,
    same_video_negative_mining: str = "random",
    feature_store: FourSlotFeatureStore | None = None,
    question_store: QuestionEmbeddingStore | None = None,
) -> list[dict[str, Any]]:
    if ranking:
        examples = _build_ranking_examples(records, node_source, split=split)
    else:
        use_random_same_video = same_video_negative_mining == "random"
        examples = _build_split_examples(
            records,
            node_source,
            split=split,
            trusted_negatives=trusted_negatives,
            seed=seed,
            same_video_negatives=(same_video_negatives if use_random_same_video else 0),
        )
        if same_video_negatives and not use_random_same_video:
            if feature_store is None or question_store is None:
                raise ValueError("visual-hard mining requires feature and question stores")
            _replace_with_visual_hard_negatives(
                examples,
                records,
                feature_store=feature_store,
                question_store=question_store,
                count=same_video_negatives,
            )
    by_case = {str(record["case_id"]): record for record in records}
    for example in examples:
        record = by_case[example["case_id"]]
        masks: list[list[bool]] = []
        for group in record.get("positive_node_groups") or ():
            nodes = set(group.get("node_ids") or ())
            mask = [node_id in nodes for node_id in example["candidate_node_ids"]]
            if any(mask):
                masks.append(mask)
        if not masks:
            raise ValueError(f"case {example['case_id']} has no covered clue group")
        example["positive_group_masks"] = masks
    return examples


def _replace_with_visual_hard_negatives(
    examples: Sequence[dict[str, Any]],
    records: Sequence[Mapping[str, Any]],
    *,
    feature_store: FourSlotFeatureStore,
    question_store: QuestionEmbeddingStore,
    count: int,
) -> None:
    """Replace cross-video negatives with highest-anchor safe same-video nodes."""

    by_case = {str(record["case_id"]): record for record in records}
    for example in examples:
        record = by_case[example["case_id"]]
        positives = [
            node_id
            for node_id, is_positive in zip(
                example["candidate_node_ids"], example["positive_mask"]
            )
            if is_positive
        ]
        cross_negatives = [
            node_id
            for node_id, is_negative in zip(
                example["candidate_node_ids"], example["negative_mask"]
            )
            if is_negative
        ]
        question = np.asarray(question_store.case(example["case_id"]), dtype=np.float32)
        question = question / max(float(np.linalg.norm(question)), 1e-12)
        ranked: list[tuple[float, str]] = []
        for node_id in record.get("trusted_same_video_negative_node_ids") or ():
            if node_id not in feature_store:
                continue
            row = feature_store.node(node_id)
            if not bool(row["validity"][2]):
                continue
            visual = np.asarray(row["features"]["visual"], dtype=np.float32)
            visual = visual / max(float(np.linalg.norm(visual)), 1e-12)
            ranked.append((float(visual @ question), node_id))
        ranked.sort(key=lambda row: (-row[0], row[1]))
        hard = [node_id for _, node_id in ranked[:count]]
        retained_cross = cross_negatives[: max(len(cross_negatives) - len(hard), 0)]
        negatives = hard + retained_cross
        candidates = positives + negatives
        example["candidate_node_ids"] = candidates
        example["positive_mask"] = [True] * len(positives) + [False] * len(negatives)
        example["negative_mask"] = [False] * len(positives) + [True] * len(negatives)


def _partition_train_records(
    records: Sequence[Mapping[str, Any]],
    *,
    fraction: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Make a deterministic video-disjoint fit/selection partition."""

    if not 0.0 < fraction < 1.0:
        raise ValueError("selection fraction must be between zero and one")
    eligible = [row for row in records if str(row.get("split") or "") == "train"]
    videos = sorted({str(row["video_id"]) for row in eligible})
    if len(videos) < 2:
        raise ValueError("fit/selection partition requires at least two train videos")
    rng = np.random.default_rng(seed)
    rng.shuffle(videos)
    selection_count = min(max(int(round(len(videos) * fraction)), 1), len(videos) - 1)
    selection_videos = set(videos[:selection_count])
    fit: list[dict[str, Any]] = []
    selection: list[dict[str, Any]] = []
    for source in eligible:
        row = dict(source)
        if str(row["video_id"]) in selection_videos:
            row["split"] = "selection"
            selection.append(row)
        else:
            row["split"] = "fit"
            fit.append(row)
    if not fit or not selection:
        raise ValueError("fit/selection partition produced an empty side")
    return fit, selection


def _initialize_qf2_from_qf1(
    modules: Mapping[str, nn.Module],
    checkpoint_path: Path,
    *,
    expected_layers: int,
    expected_hidden_size: int,
) -> None:
    """BLIP-style transfer from feature-fusion pretraining into conditioned QF2."""

    checkpoint = torch.load(
        checkpoint_path.expanduser().resolve(), map_location="cpu", weights_only=True
    )
    config = checkpoint.get("config") or {}
    if int(config.get("num_layers", -1)) != expected_layers:
        raise ValueError("QF2 layers must match the QF1 initialization checkpoint")
    if int(config.get("hidden_size", -1)) != expected_hidden_size:
        raise ValueError("QF2 hidden size disagrees with QF1 initialization checkpoint")
    modules["qf2"].load_state_dict(checkpoint["qf1"], strict=True)
    projector = checkpoint["projector"]
    question_state = {
        "0.weight": projector["projectors.caption.0.weight"],
        "0.bias": projector["projectors.caption.0.bias"],
        "1.weight": projector["projectors.caption.1.weight"],
        "1.bias": projector["projectors.caption.1.bias"],
    }
    modules["qf_question"].load_state_dict(question_state, strict=True)
    scorer = modules["qf_scorer"]
    assert isinstance(scorer, NodeScoreHead)
    with torch.no_grad():
        nn.init.eye_(scorer.proposal_projection.weight)
        nn.init.eye_(scorer.question_projection.weight)


def _forward(modules, batch, device):
    question = batch["question"].to(device)
    semantic = batch["semantic"].to(device)
    time_values = batch["time"].to(device)
    qf_score = None
    if "qf2" in modules:
        nodes = batch["qf_nodes"].to(device)
        batch_size, node_count, memory_count, hidden_size = nodes.shape
        qf_question = modules["qf_question"](question).unsqueeze(1)
        repeated = qf_question[:, None].expand(-1, node_count, -1, -1).reshape(
            batch_size * node_count, 1, hidden_size
        )
        proposals = modules["qf2"](
            nodes.reshape(batch_size * node_count, memory_count, hidden_size), repeated
        ).reshape(batch_size, node_count, 8, hidden_size)
        qf_score = modules["qf_scorer"](proposals, qf_question)
    scores, diagnostics = modules["residual"](
        question, semantic, time_values, qf_score=qf_score
    )
    return (
        scores,
        batch["groups"].to(device),
        batch["group_validity"].to(device),
        batch["negative"].to(device),
        diagnostics,
    )


@torch.no_grad()
def _evaluate_trusted(modules, loader, device):
    for module in modules.values():
        module.eval()
    total = 0.0
    count = 0
    for batch in loader:
        scores, groups, group_validity, negatives, _ = _forward(modules, batch, device)
        loss = trusted_group_retrieval_loss(scores, groups, group_validity, negatives)
        group_count = int(group_validity.sum())
        total += float(loss) * group_count
        count += group_count
    return {"group_balanced_loss": total / max(count, 1)}


@torch.no_grad()
def _evaluate_ranking(modules, loader, device, *, include_per_case: bool = False):
    for module in modules.values():
        module.eval()
    score_rows: list[np.ndarray] = []
    positive_rows: list[np.ndarray] = []
    valid_rows: list[np.ndarray] = []
    group_rows: list[np.ndarray] = []
    permutation_errors: list[dict[str, float]] = []
    gate_rows: list[np.ndarray] = []
    case_ids: list[str] = []
    last_diagnostics = None
    for batch in loader:
        case_ids.append(str(batch["case_ids"][0]))
        scores, groups, group_validity, _, diagnostics = _forward(modules, batch, device)
        valid = batch["positive"] | batch["negative"]
        node_count = int(valid[0].sum())
        group_count = int(group_validity[0].sum())
        score_rows.append(scores[0, :node_count].cpu().numpy())
        positive_rows.append(batch["positive"][0, :node_count].numpy())
        valid_rows.append(valid[0, :node_count].numpy())
        group_rows.append(groups[0, :group_count, :node_count].cpu().numpy())
        gate_rows.append(diagnostics["gates"].cpu().numpy())
        last_diagnostics = diagnostics
        order = torch.arange(node_count - 1, -1, -1)
        permuted = dict(batch)
        for name in ("semantic", "time", "positive", "negative"):
            permuted[name] = batch[name][:, order]
        permuted["groups"] = batch["groups"][:, :, order]
        if "qf_nodes" in batch:
            permuted["qf_nodes"] = batch["qf_nodes"][:, order]
        permuted_scores, _, _, _, _ = _forward(modules, permuted, device)
        permutation_errors.append(
            permutation_consistency(
                scores[:, :node_count].cpu().numpy(),
                permuted_scores[:, :node_count].cpu().numpy(),
                np.tile(order.numpy(), (scores.shape[0], 1)),
            )
        )
    width = max(len(row) for row in score_rows)
    score_matrix = np.full((len(score_rows), width), -np.inf, dtype=np.float32)
    positive_matrix = np.zeros((len(score_rows), width), dtype=np.bool_)
    valid_matrix = np.zeros_like(positive_matrix)
    padded_groups: list[np.ndarray] = []
    for index, (scores, positives, valid, groups) in enumerate(
        zip(score_rows, positive_rows, valid_rows, group_rows)
    ):
        score_matrix[index, : len(scores)] = scores
        positive_matrix[index, : len(positives)] = positives
        valid_matrix[index, : len(valid)] = valid
        padded = np.zeros((len(groups), width), dtype=np.bool_)
        padded[:, : groups.shape[1]] = groups
        padded_groups.append(padded)
    assert last_diagnostics is not None
    report = {
        "question_count": len(score_rows),
        "node_union_metrics": retrieval_metrics(
            score_matrix, positive_matrix, valid_matrix
        ),
        "clue_group_metrics": clue_group_metrics(
            score_matrix, valid_matrix, padded_groups
        ),
        "permutation": {
            "max_absolute_error": max(
                row["max_absolute_error"] for row in permutation_errors
            ),
            "mean_absolute_error": float(
                np.mean([row["mean_absolute_error"] for row in permutation_errors])
            ),
        },
        "mean_question_gates": np.concatenate(gate_rows, axis=0).mean(axis=0).tolist(),
        "learned_scales": {
            "caption": float(last_diagnostics["semantic_scales"][0]),
            "entity": float(last_diagnostics["semantic_scales"][1]),
            "time": float(last_diagnostics["time_scale"][0]),
            "qf": float(last_diagnostics["qf_scale"][0]),
            "visual_anchor": float(last_diagnostics["anchor_scale"][0]),
        },
    }
    if include_per_case:
        report["per_case"] = []
        for index, case_id in enumerate(case_ids):
            report["per_case"].append(
                {
                    "case_id": case_id,
                    "node_union_metrics": retrieval_metrics(
                        score_matrix[index : index + 1],
                        positive_matrix[index : index + 1],
                        valid_matrix[index : index + 1],
                    ),
                    "clue_group_metrics": clue_group_metrics(
                        score_matrix[index : index + 1],
                        valid_matrix[index : index + 1],
                        [padded_groups[index]],
                    ),
                }
            )
    return report


def _selection_key(report: Mapping[str, Any]) -> tuple[float, float, float]:
    groups = report["clue_group_metrics"]
    union = report["node_union_metrics"]
    return (
        float(groups["clue_group_recall@8"]),
        float(groups["all_clue_group_coverage@8"]),
        float(union["mrr"]),
    )


def _save_checkpoint(path, modules, variant, epoch, ranking):
    torch.save(
        {
            "schema_version": "steam-qformer-gated-residual-checkpoint/v0.2",
            "variant": variant,
            "epoch": epoch,
            "selection_key": _selection_key(ranking),
            "modules": {name: module.state_dict() for name, module in modules.items()},
        },
        path,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=tuple(VARIANTS), required=True)
    parser.add_argument("--train-feature-manifest", required=True, type=Path)
    parser.add_argument("--validation-feature-manifest", required=True, type=Path)
    parser.add_argument("--train-labels", required=True, type=Path)
    parser.add_argument("--validation-labels", required=True, type=Path)
    parser.add_argument("--train-question-manifest", required=True, type=Path)
    parser.add_argument("--validation-question-manifest", required=True, type=Path)
    parser.add_argument("--train-qf-cache-manifest", type=Path)
    parser.add_argument("--validation-qf-cache-manifest", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--num-heads", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--trusted-negatives", type=int, default=32)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--selection-fraction", type=float, default=0.2)
    parser.add_argument("--selection-seed", type=int, default=1729)
    parser.add_argument("--same-video-negatives", type=int, default=0)
    parser.add_argument(
        "--same-video-negative-mining",
        choices=("random", "visual_hard"),
        default="random",
    )
    parser.add_argument("--qf2-init-checkpoint", type=Path)
    args = parser.parse_args(argv)
    report = train_gated_residual(
        variant=args.variant,
        train_feature_manifest=args.train_feature_manifest,
        validation_feature_manifest=args.validation_feature_manifest,
        train_labels_path=args.train_labels,
        validation_labels_path=args.validation_labels,
        train_question_manifest=args.train_question_manifest,
        validation_question_manifest=args.validation_question_manifest,
        train_qf_cache_manifest=args.train_qf_cache_manifest,
        validation_qf_cache_manifest=args.validation_qf_cache_manifest,
        output_dir=args.output_dir,
        device=args.device,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        trusted_negatives=args.trusted_negatives,
        seed=args.seed,
        selection_fraction=args.selection_fraction,
        selection_seed=args.selection_seed,
        same_video_negatives=args.same_video_negatives,
        same_video_negative_mining=args.same_video_negative_mining,
        qf2_init_checkpoint=args.qf2_init_checkpoint,
    )
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
