"""Persistence and validation for four-slot Q-Former feature sidecars."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .contracts import (
    SLOT_NAMES,
    FeatureMatrixSpec,
    FeatureRow,
    FeatureStoreManifest,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class FourSlotFeatureStore:
    """Read-only validated view over separately persisted slot matrices."""

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        verify_checksums: bool = True,
        mmap_mode: str | None = "r",
    ) -> None:
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.manifest = FeatureStoreManifest.from_dict(payload)
        self._arrays = {
            name: self._load(spec, verify_checksums, mmap_mode)
            for name, spec in self.manifest.matrices.items()
        }
        self._validity = self._load(
            self.manifest.validity, verify_checksums, mmap_mode
        )
        self._validate_shapes()
        self._row_by_node = {row.node_id: row for row in self.manifest.rows}

    def _load(
        self,
        spec: FeatureMatrixSpec,
        verify_checksums: bool,
        mmap_mode: str | None,
    ) -> np.ndarray:
        path = self.manifest.resolve(self.manifest_path, spec.path)
        if not path.is_file():
            raise FileNotFoundError(path)
        if verify_checksums and sha256_file(path) != spec.checksum:
            raise ValueError(f"feature checksum mismatch: {path}")
        array = np.load(path, mmap_mode=mmap_mode, allow_pickle=False)
        if str(array.dtype) != spec.dtype:
            raise ValueError(f"feature dtype mismatch for {path}: {array.dtype}")
        return array

    def _validate_shapes(self) -> None:
        count = len(self.manifest.rows)
        for name in SLOT_NAMES:
            array = self._arrays[name]
            spec = self.manifest.matrices[name]
            if array.shape != (count, spec.dimension):
                raise ValueError(
                    f"{name} matrix has shape {array.shape}; "
                    f"expected {(count, spec.dimension)}"
                )
            if not np.isfinite(array).all():
                raise ValueError(f"{name} matrix contains non-finite values")
        if self._validity.shape != (count, len(SLOT_NAMES)):
            raise ValueError(
                f"validity matrix has shape {self._validity.shape}; "
                f"expected {(count, len(SLOT_NAMES))}"
            )
        if self._validity.dtype != np.bool_:
            raise ValueError("validity matrix must use bool dtype")
        if count and (~self._validity.any(axis=1)).any():
            raise ValueError("feature store contains a node with no valid slots")

    def __len__(self) -> int:
        return len(self.manifest.rows)

    def __contains__(self, node_id: str) -> bool:
        return node_id in self._row_by_node

    def row(self, index: int) -> dict[str, Any]:
        metadata = self.manifest.rows[index]
        return {
            "node_id": metadata.node_id,
            "video_id": metadata.video_id,
            "lineage_hash": metadata.lineage_hash,
            "features": {name: self._arrays[name][index] for name in SLOT_NAMES},
            "validity": self._validity[index],
        }

    def node(self, node_id: str) -> dict[str, Any]:
        try:
            return self.row(self._row_by_node[node_id].row_index)
        except KeyError as exc:
            raise KeyError(f"unknown feature node: {node_id}") from exc

    def coverage(self) -> dict[str, float]:
        if not len(self):
            return {name: 0.0 for name in SLOT_NAMES}
        return {
            name: float(self._validity[:, index].mean())
            for index, name in enumerate(SLOT_NAMES)
        }


def write_feature_store(
    destination: str | Path,
    *,
    arrays: Mapping[str, np.ndarray],
    validity: np.ndarray,
    rows: Sequence[FeatureRow],
    encoders: Mapping[str, str],
    source_contract: str,
    boundary_audit_version: str,
    normalized: Mapping[str, bool] | None = None,
) -> Path:
    """Write one immutable feature store and return its manifest path."""

    destination = Path(destination).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    if set(arrays) != set(SLOT_NAMES):
        raise ValueError(f"arrays must contain exactly {SLOT_NAMES}")
    if set(encoders) != set(SLOT_NAMES):
        raise ValueError(f"encoders must contain exactly {SLOT_NAMES}")
    normalized = normalized or {name: False for name in SLOT_NAMES}
    if set(normalized) != set(SLOT_NAMES):
        raise ValueError(f"normalized must contain exactly {SLOT_NAMES}")
    if validity.shape != (len(rows), len(SLOT_NAMES)):
        raise ValueError("validity shape disagrees with rows")
    if validity.dtype != np.bool_:
        validity = validity.astype(np.bool_, copy=False)

    specs: dict[str, FeatureMatrixSpec] = {}
    for name in SLOT_NAMES:
        matrix = np.asarray(arrays[name])
        if matrix.ndim != 2 or matrix.shape[0] != len(rows):
            raise ValueError(f"{name} must have shape [N, D]")
        if not np.isfinite(matrix).all():
            raise ValueError(f"{name} contains non-finite values")
        path = destination / f"{name}.npy"
        np.save(path, matrix, allow_pickle=False)
        specs[name] = FeatureMatrixSpec(
            path=path.name,
            dimension=int(matrix.shape[1]),
            dtype=str(matrix.dtype),
            checksum=sha256_file(path),
            encoder=str(encoders[name]),
            normalized=bool(normalized[name]),
        )

    validity_path = destination / "validity.npy"
    np.save(validity_path, validity, allow_pickle=False)
    validity_spec = FeatureMatrixSpec(
        path=validity_path.name,
        dimension=len(SLOT_NAMES),
        dtype=str(validity.dtype),
        checksum=sha256_file(validity_path),
        encoder="deterministic-validity-mask/v1",
        normalized=False,
    )
    manifest = FeatureStoreManifest(
        matrices=specs,
        validity=validity_spec,
        rows=tuple(rows),
        source_contract=source_contract,
        boundary_audit_version=boundary_audit_version,
    )
    manifest_path = destination / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    FourSlotFeatureStore(manifest_path)
    return manifest_path
