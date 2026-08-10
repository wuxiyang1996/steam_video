from __future__ import annotations

import pytest
import numpy as np


torch = pytest.importorskip("torch")

from steam_video_new.implicit_world_model.reasoning_v2.qformer.contracts import SLOT_NAMES
from steam_video_new.implicit_world_model.reasoning_v2.qformer.losses import (
    masked_heterogeneous_slot_distillation_loss,
    masked_slot_distillation_loss,
    trusted_multi_positive_retrieval_loss,
)
from steam_video_new.implicit_world_model.reasoning_v2.qformer.model import (
    ConditionedProposalQFormer,
    FourSlotDecoder,
    FourSlotProjector,
    HeterogeneousSlotHeads,
    IndependentNodeQFormer,
    NodeScoreHead,
)
from steam_video_new.implicit_world_model.reasoning_v2.qformer import (
    FeatureRow,
    FourSlotFeatureStore,
    QF1Cache,
    write_feature_store,
)
from steam_video_new.implicit_world_model.reasoning_v2.qformer.export_qf1 import (
    export_qf1_cache,
    export_projected_slots,
)


def _models():
    dimensions = {name: index + 3 for index, name in enumerate(SLOT_NAMES)}
    projector = FourSlotProjector(dimensions, 16)
    qf1 = IndependentNodeQFormer(hidden_size=16, num_layers=2, num_heads=4)
    decoder = FourSlotDecoder(hidden_size=16, num_heads=4)
    heads = HeterogeneousSlotHeads(dimensions, 16)
    qf2 = ConditionedProposalQFormer(hidden_size=16, num_layers=2, num_heads=4)
    return dimensions, projector, qf1, decoder, heads, qf2


def test_dual_qformer_shapes_and_independent_queries() -> None:
    dimensions, projector, qf1, decoder, heads, qf2 = _models()
    features = {name: torch.randn(3, dimension) for name, dimension in dimensions.items()}
    validity = torch.tensor([[1, 1, 1, 1], [1, 0, 1, 1], [1, 1, 0, 1]]).bool()
    typed = projector(features, validity)
    u = qf1(typed, validity)
    reconstructed = decoder(u)
    raw_predictions = heads(reconstructed)
    question = torch.randn(3, 2, 16)
    p = qf2(u, question)
    assert typed.shape == (3, 4, 16)
    assert u.shape == (3, 8, 16)
    assert reconstructed.shape == (3, 4, 16)
    assert {name: value.shape for name, value in raw_predictions.items()} == {
        name: (3, dimension) for name, dimension in dimensions.items()
    }
    assert p.shape == (3, 8, 16)
    assert qf1.query_tokens.data_ptr() != qf2.query_tokens.data_ptr()


def test_slot_loss_ignores_invalid_target() -> None:
    predicted = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    target = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])
    validity = torch.tensor([[True, False]])
    assert masked_slot_distillation_loss(predicted, target, validity).item() == pytest.approx(0.0)


def test_heterogeneous_slot_loss_reconstructs_frozen_source_dimensions() -> None:
    targets = {
        name: torch.randn(2, index + 2) for index, name in enumerate(SLOT_NAMES)
    }
    predictions = {name: value.clone().requires_grad_() for name, value in targets.items()}
    validity = torch.tensor([[1, 1, 1, 1], [1, 0, 1, 1]]).bool()
    loss = masked_heterogeneous_slot_distillation_loss(
        predictions, targets, validity
    )
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_retrieval_loss_ignores_unlabeled_and_accepts_multiple_positives() -> None:
    scores = torch.tensor([[3.0, 2.0, -100.0, 0.0]], requires_grad=True)
    positives = torch.tensor([[True, True, False, False]])
    negatives = torch.tensor([[False, False, False, True]])
    loss = trusted_multi_positive_retrieval_loss(scores, positives, negatives)
    loss.backward()
    assert scores.grad[0, 2].item() == 0.0
    assert scores.grad[0, 0].item() < 0.0
    assert scores.grad[0, 1].item() < 0.0
    assert scores.grad[0, 3].item() > 0.0


def test_qf1_can_be_frozen_while_qf2_receives_gradients() -> None:
    _, _, qf1, _, _, qf2 = _models()
    for parameter in qf1.parameters():
        parameter.requires_grad_(False)
    u = torch.randn(2, 8, 16)
    p = qf2(u, torch.randn(2, 1, 16))
    p.square().mean().backward()
    assert all(parameter.grad is None for parameter in qf1.parameters())
    assert any(parameter.grad is not None for parameter in qf2.parameters())


def test_node_score_shapes() -> None:
    scorer = NodeScoreHead(16)
    scores = scorer(torch.randn(2, 5, 8, 16), torch.randn(2, 3, 16))
    assert scores.shape == (2, 5)


def test_qf1_cache_export_round_trip_and_lineage(tmp_path) -> None:
    dimensions, projector, qf1, _, _, _ = _models()
    rows = tuple(
        FeatureRow(index, f"node:{index}", f"video:{index // 2}", f"hash:{index}")
        for index in range(4)
    )
    feature_manifest = write_feature_store(
        tmp_path / "features",
        arrays={
            name: np.random.default_rng(index).normal(size=(4, dimension)).astype(np.float32)
            for index, (name, dimension) in enumerate(dimensions.items())
        },
        validity=np.ones((4, len(SLOT_NAMES)), dtype=np.bool_),
        rows=rows,
        encoders={name: f"fixture/{name}" for name in SLOT_NAMES},
        source_contract="fixture-safe-pre-read/v1",
        boundary_audit_version="fixture-boundary/v1",
    )
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save({"fixture": True}, checkpoint)
    report = export_qf1_cache(
        projector=projector,
        qf1=qf1,
        store=FourSlotFeatureStore(feature_manifest),
        checkpoint_path=checkpoint,
        destination=tmp_path / "cache",
    )
    cache = QF1Cache(report["manifest"], feature_manifest_path=feature_manifest)
    assert len(cache) == 4
    assert cache.node("node:2").shape == (8, 16)


def test_projector_only_cache_has_four_memory_tokens(tmp_path) -> None:
    dimensions, projector, _, _, _, _ = _models()
    rows = tuple(
        FeatureRow(index, f"node:{index}", f"video:{index // 2}", f"hash:{index}")
        for index in range(4)
    )
    feature_manifest = write_feature_store(
        tmp_path / "features",
        arrays={
            name: np.random.default_rng(index).normal(size=(4, dimension)).astype(np.float32)
            for index, (name, dimension) in enumerate(dimensions.items())
        },
        validity=np.ones((4, len(SLOT_NAMES)), dtype=np.bool_),
        rows=rows,
        encoders={name: f"fixture/{name}" for name in SLOT_NAMES},
        source_contract="fixture-safe-pre-read/v1",
        boundary_audit_version="fixture-boundary/v1",
    )
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "config": {"slot_dimensions": dimensions, "hidden_size": 16},
            "projector": projector.state_dict(),
            "feature_manifest": str(feature_manifest),
        },
        checkpoint,
    )
    report = export_projected_slots(
        checkpoint,
        tmp_path / "projected",
        feature_manifest=feature_manifest,
    )
    cache = QF1Cache(report["manifest"], feature_manifest_path=feature_manifest)
    assert cache.node("node:1").shape == (4, 16)
