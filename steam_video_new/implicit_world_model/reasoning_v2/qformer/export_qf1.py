"""Export frozen QF1 node slots with checkpoint and feature lineage checksums."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from .contracts import FeatureMatrixSpec, QF1CacheManifest
from .datasets import QF1FeatureDataset, collate_qf1
from .feature_store import FourSlotFeatureStore, sha256_file
from .model import FourSlotProjector, IndependentNodeQFormer


@torch.no_grad()
def export_qf1_cache(
    *,
    projector: torch.nn.Module,
    qf1: torch.nn.Module,
    store: FourSlotFeatureStore,
    checkpoint_path: Path,
    destination: Path,
    batch_size: int = 128,
    device: str = "cpu",
) -> dict[str, Any]:
    checkpoint_path = checkpoint_path.expanduser().resolve()
    destination = destination.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    projector.eval().to(device)
    qf1.eval().to(device)
    outputs: list[np.ndarray] = []
    loader = DataLoader(
        QF1FeatureDataset(store),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_qf1,
    )
    for batch in loader:
        features = {name: value.to(device) for name, value in batch["features"].items()}
        validity = batch["validity"].to(device)
        typed = projector(features, validity)
        outputs.append(qf1(typed, validity).cpu().numpy().astype(np.float32))
    matrix = np.concatenate(outputs, axis=0)
    matrix_path = destination / "qf1_tokens.npy"
    np.save(matrix_path, matrix, allow_pickle=False)
    feature_manifest_checksum = sha256_file(store.manifest_path)
    checkpoint_checksum = sha256_file(checkpoint_path)
    spec = FeatureMatrixSpec(
        path=matrix_path.name,
        dimension=int(matrix.shape[1] * matrix.shape[2]),
        dtype=str(matrix.dtype),
        checksum=sha256_file(matrix_path),
        encoder=f"qf1-checkpoint:{checkpoint_checksum}",
    )
    manifest = QF1CacheManifest(
        checkpoint_checksum=checkpoint_checksum,
        feature_manifest_checksum=feature_manifest_checksum,
        matrix=spec,
        rows=store.manifest.rows,
        num_queries=int(matrix.shape[1]),
        hidden_size=int(matrix.shape[2]),
    )
    manifest_path = destination / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest.to_dict(), indent=2)
        + "\n",
        encoding="utf-8",
    )
    return {
        "manifest": str(manifest_path),
        "shape": list(matrix.shape),
        "checkpoint_checksum": checkpoint_checksum,
        "feature_manifest_checksum": feature_manifest_checksum,
    }


def export_checkpoint(
    checkpoint_path: Path,
    destination: Path,
    *,
    feature_manifest: Path | None = None,
    batch_size: int = 128,
    device: str = "cpu",
) -> dict[str, Any]:
    checkpoint_path = checkpoint_path.expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    manifest_path = Path(
        feature_manifest or checkpoint.get("feature_manifest") or ""
    ).expanduser().resolve()
    store = FourSlotFeatureStore(manifest_path)
    projector = FourSlotProjector(
        config["slot_dimensions"], int(config["hidden_size"])
    )
    qf1 = IndependentNodeQFormer(
        hidden_size=int(config["hidden_size"]),
        num_queries=int(config["num_queries"]),
        num_layers=int(config["num_layers"]),
        num_heads=int(config["num_heads"]),
    )
    projector.load_state_dict(checkpoint["projector"])
    qf1.load_state_dict(checkpoint["qf1"])
    return export_qf1_cache(
        projector=projector,
        qf1=qf1,
        store=store,
        checkpoint_path=checkpoint_path,
        destination=destination,
        batch_size=batch_size,
        device=device,
    )


@torch.no_grad()
def export_projected_slots(
    checkpoint_path: Path,
    destination: Path,
    *,
    feature_manifest: Path | None = None,
    batch_size: int = 128,
    device: str = "cpu",
) -> dict[str, Any]:
    """Export four typed projector slots for the matched B2 ablation."""

    checkpoint_path = checkpoint_path.expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    manifest_path = Path(
        feature_manifest or checkpoint.get("feature_manifest") or ""
    ).expanduser().resolve()
    store = FourSlotFeatureStore(manifest_path)
    projector = FourSlotProjector(
        config["slot_dimensions"], int(config["hidden_size"])
    )
    projector.load_state_dict(checkpoint["projector"])
    projector.eval().to(device)
    outputs: list[np.ndarray] = []
    loader = DataLoader(
        QF1FeatureDataset(store),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_qf1,
    )
    for batch in loader:
        features = {name: value.to(device) for name, value in batch["features"].items()}
        validity = batch["validity"].to(device)
        outputs.append(projector(features, validity).cpu().numpy().astype(np.float32))
    matrix = np.concatenate(outputs, axis=0)
    destination = destination.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    matrix_path = destination / "projected_slots.npy"
    np.save(matrix_path, matrix, allow_pickle=False)
    checkpoint_checksum = sha256_file(checkpoint_path)
    feature_manifest_checksum = sha256_file(store.manifest_path)
    spec = FeatureMatrixSpec(
        path=matrix_path.name,
        dimension=int(matrix.shape[1] * matrix.shape[2]),
        dtype=str(matrix.dtype),
        checksum=sha256_file(matrix_path),
        encoder=f"qf1-checkpoint-projector-only:{checkpoint_checksum}",
    )
    manifest = QF1CacheManifest(
        checkpoint_checksum=checkpoint_checksum,
        feature_manifest_checksum=feature_manifest_checksum,
        matrix=spec,
        rows=store.manifest.rows,
        num_queries=int(matrix.shape[1]),
        hidden_size=int(matrix.shape[2]),
    )
    manifest_path = destination / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest.to_dict(), indent=2) + "\n", encoding="utf-8"
    )
    return {
        "manifest": str(manifest_path),
        "shape": list(matrix.shape),
        "model_variant": "b2_four_projected_slots_without_qf1_fusion",
        "checkpoint_checksum": checkpoint_checksum,
        "feature_manifest_checksum": feature_manifest_checksum,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--feature-manifest", type=Path)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--projector-only", action="store_true")
    args = parser.parse_args(argv)
    function = export_projected_slots if args.projector_only else export_checkpoint
    report = function(
        args.checkpoint,
        args.destination,
        feature_manifest=args.feature_manifest,
        batch_size=args.batch_size,
        device=args.device,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
