"""Materialize already-computed, boundary-approved four-slot vectors from JSONL."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .contracts import SLOT_NAMES, FeatureRow
from .feature_store import FourSlotFeatureStore, write_feature_store


def materialize_jsonl(
    source: Path,
    destination: Path,
    *,
    source_contract: str,
    boundary_audit_version: str,
) -> dict[str, Any]:
    """Build a feature store without deriving data from hidden EvidenceValue fields."""

    records = [
        json.loads(line)
        for line in source.expanduser().resolve().read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        raise ValueError("feature JSONL is empty")
    dimensions: dict[str, int] = {}
    values: dict[str, list[list[float]]] = {name: [] for name in SLOT_NAMES}
    masks: list[list[bool]] = []
    rows: list[FeatureRow] = []
    encoders: dict[str, str] = {}
    for index, record in enumerate(records):
        node_id = str(record.get("node_id") or "")
        video_id = str(record.get("video_id") or "")
        lineage_hash = str(record.get("lineage_hash") or _record_hash(record))
        slots = record.get("slots") or {}
        mask: list[bool] = []
        for name in SLOT_NAMES:
            payload = slots.get(name) or {}
            vector = payload.get("vector")
            valid = bool(payload.get("valid", vector is not None))
            if vector is None:
                dimension = int(payload.get("dimension") or dimensions.get(name) or 0)
                if dimension <= 0:
                    raise ValueError(
                        f"record {index} missing {name} dimension before it is established"
                    )
                vector = [0.0] * dimension
            vector = [float(value) for value in vector]
            dimensions.setdefault(name, len(vector))
            if len(vector) != dimensions[name]:
                raise ValueError(f"record {index} has inconsistent {name} dimension")
            encoder = str(payload.get("encoder") or encoders.get(name) or "")
            if not encoder:
                raise ValueError(f"record {index} lacks encoder for {name}")
            if name in encoders and encoder != encoders[name]:
                raise ValueError(f"record {index} changes encoder for {name}")
            encoders[name] = encoder
            values[name].append(vector)
            mask.append(valid)
        if not any(mask):
            raise ValueError(f"record {index} has no valid feature slots")
        masks.append(mask)
        rows.append(FeatureRow(index, node_id, video_id, lineage_hash))

    manifest_path = write_feature_store(
        destination,
        arrays={name: np.asarray(values[name], dtype=np.float32) for name in SLOT_NAMES},
        validity=np.asarray(masks, dtype=np.bool_),
        rows=rows,
        encoders=encoders,
        source_contract=source_contract,
        boundary_audit_version=boundary_audit_version,
    )
    store = FourSlotFeatureStore(manifest_path)
    return {
        "manifest": str(manifest_path),
        "node_count": len(store),
        "video_count": len({row.video_id for row in store.manifest.rows}),
        "coverage": store.coverage(),
        "complete_four_slot_count": sum(
            bool(store.row(index)["validity"].all()) for index in range(len(store))
        ),
    }


def _record_hash(record: dict[str, Any]) -> str:
    payload = {
        "node_id": record.get("node_id"),
        "video_id": record.get("video_id"),
        "source_refs": record.get("source_refs"),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--source-contract", required=True)
    parser.add_argument("--boundary-audit-version", required=True)
    args = parser.parse_args(argv)
    report = materialize_jsonl(
        args.input_jsonl,
        args.output_dir,
        source_contract=args.source_contract,
        boundary_audit_version=args.boundary_audit_version,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

