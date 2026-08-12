"""Train QF1 with one masked reconstruction objective over frozen raw slots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from .contracts import SLOT_NAMES
from .datasets import QF1FeatureDataset, collate_qf1
from .feature_store import FourSlotFeatureStore
from .losses import masked_heterogeneous_slot_distillation_loss
from .model import (
    FourSlotDecoder,
    FourSlotProjector,
    HeterogeneousSlotHeads,
    IndependentNodeQFormer,
)
from .splits import video_disjoint_indices


def train_qf1(
    manifest_path: Path,
    output_dir: Path,
    *,
    device: str,
    hidden_size: int = 768,
    num_queries: int = 8,
    num_layers: int = 4,
    num_heads: int = 12,
    batch_size: int = 64,
    epochs: int = 5,
    learning_rate: float = 1e-4,
    validation_fraction: float = 0.2,
    seed: int = 17,
) -> dict[str, Any]:
    _seed(seed)
    store = FourSlotFeatureStore(manifest_path)
    if len({row.video_id for row in store.manifest.rows}) < 2:
        raise ValueError("formal QF1 training requires at least two videos")
    train_indices, validation_indices = video_disjoint_indices(
        [row.video_id for row in store.manifest.rows],
        validation_fraction=validation_fraction,
    )
    train_loader = DataLoader(
        QF1FeatureDataset(store, train_indices),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_qf1,
    )
    validation_loader = DataLoader(
        QF1FeatureDataset(store, validation_indices),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_qf1,
    )
    dimensions = {
        name: store.manifest.matrices[name].dimension for name in SLOT_NAMES
    }
    target_device = torch.device(device)
    projector = FourSlotProjector(dimensions, hidden_size).to(target_device)
    qf1 = IndependentNodeQFormer(
        hidden_size=hidden_size,
        num_queries=num_queries,
        num_layers=num_layers,
        num_heads=num_heads,
    ).to(target_device)
    decoder = FourSlotDecoder(
        hidden_size=hidden_size,
        num_layers=1,
        num_heads=num_heads,
    ).to(target_device)
    heads = HeterogeneousSlotHeads(dimensions, hidden_size).to(target_device)
    modules = (projector, qf1, decoder, heads)
    optimizer = torch.optim.AdamW(
        [parameter for module in modules for parameter in module.parameters()],
        lr=learning_rate,
        weight_decay=0.05,
    )
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, Any]] = []
    initial_validation = _evaluate(
        modules, validation_loader, target_device
    )
    best_loss = float("inf")
    checkpoint_path = output_dir / "qf1_best.pt"
    for epoch in range(1, epochs + 1):
        for module in modules:
            module.train()
        total = 0.0
        count = 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss, _ = _forward(modules, batch, target_device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [parameter for module in modules for parameter in module.parameters()],
                max_norm=1.0,
            )
            optimizer.step()
            # The loss is normalized by valid slots, so aggregate epochs with
            # the same denominator when some entity/state slots are masked.
            batch_count = int(batch["validity"].sum())
            total += float(loss.detach()) * batch_count
            count += batch_count
        validation = _evaluate(modules, validation_loader, target_device)
        epoch_report = {
            "epoch": epoch,
            "train_loss": total / max(count, 1),
            "validation": validation,
        }
        history.append(epoch_report)
        if validation["loss"] < best_loss:
            best_loss = validation["loss"]
            torch.save(
                {
                    "schema_version": "steam-qformer-qf1-checkpoint/v0.1",
                    "config": {
                        "slot_dimensions": dimensions,
                        "hidden_size": hidden_size,
                        "num_queries": num_queries,
                        "num_layers": num_layers,
                        "num_heads": num_heads,
                    },
                    "projector": projector.state_dict(),
                    "qf1": qf1.state_dict(),
                    "decoder": decoder.state_dict(),
                    "slot_heads": heads.state_dict(),
                    "epoch": epoch,
                    "validation": validation,
                    "feature_manifest": str(store.manifest_path),
                },
                checkpoint_path,
            )

    report = {
        "schema_version": "steam-qformer-qf1-training/v0.1",
        "feature_manifest": str(store.manifest_path),
        "checkpoint": str(checkpoint_path),
        "config": {
            "hidden_size": hidden_size,
            "num_queries": num_queries,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "batch_size": batch_size,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "seed": seed,
        },
        "data": {
            "train_nodes": len(train_indices),
            "validation_nodes": len(validation_indices),
            "train_videos": len({store.manifest.rows[index].video_id for index in train_indices}),
            "validation_videos": len(
                {store.manifest.rows[index].video_id for index in validation_indices}
            ),
            "slot_coverage": store.coverage(),
        },
        "initial_validation": initial_validation,
        "history": history,
        "best_validation_loss": best_loss,
        "gates": {
            "video_disjoint": {store.manifest.rows[index].video_id for index in train_indices}.isdisjoint(
                {store.manifest.rows[index].video_id for index in validation_indices}
            ),
            "finite_losses": bool(
                np.isfinite(
                    [initial_validation["loss"], best_loss]
                    + [row["train_loss"] for row in history]
                ).all()
            ),
            "validation_improved": best_loss < initial_validation["loss"],
        },
    }
    report["passed"] = all(report["gates"].values())
    (output_dir / "training_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def _forward(modules, batch, device):
    projector, qf1, decoder, heads = modules
    features = {name: value.to(device) for name, value in batch["features"].items()}
    validity = batch["validity"].to(device)
    typed = projector(features, validity)
    u = qf1(typed, validity)
    predictions = heads(decoder(u))
    loss = masked_heterogeneous_slot_distillation_loss(
        predictions, features, validity
    )
    return loss, (predictions, features, validity)


@torch.no_grad()
def _evaluate(modules, loader, device):
    for module in modules:
        module.eval()
    total = 0.0
    count = 0
    cosine_sum = {name: 0.0 for name in SLOT_NAMES}
    slot_count = {name: 0 for name in SLOT_NAMES}
    for batch in loader:
        loss, (predictions, features, validity) = _forward(modules, batch, device)
        batch_count = int(validity.sum())
        total += float(loss) * batch_count
        count += batch_count
        for index, name in enumerate(SLOT_NAMES):
            cosine = torch.nn.functional.cosine_similarity(
                predictions[name], features[name], dim=-1
            )
            mask = validity[:, index]
            cosine_sum[name] += float(cosine[mask].sum())
            slot_count[name] += int(mask.sum())
    return {
        "loss": total / max(count, 1),
        "slot_cosine": {
            name: cosine_sum[name] / max(slot_count[name], 1) for name in SLOT_NAMES
        },
        "slot_count": slot_count,
    }


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--hidden-size", type=int, default=768)
    parser.add_argument("--num-queries", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args(argv)
    report = train_qf1(
        args.manifest,
        args.output_dir,
        device=args.device,
        hidden_size=args.hidden_size,
        num_queries=args.num_queries,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
    )
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
