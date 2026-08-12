"""Versioned contracts for four-slot node features and Q-Former caches."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


FEATURE_STORE_SCHEMA = "steam-qformer-feature-store/v0.1"
QF1_CACHE_SCHEMA = "steam-qformer-qf1-cache/v0.1"
SLOT_NAMES = ("caption", "entity_state", "visual", "time")


@dataclass(frozen=True)
class FeatureMatrixSpec:
    path: str
    dimension: int
    dtype: str
    checksum: str
    encoder: str
    normalized: bool = False

    def __post_init__(self) -> None:
        if not self.path or not self.checksum or not self.encoder:
            raise ValueError("feature matrix spec is incomplete")
        if self.dimension <= 0:
            raise ValueError("feature matrix dimension must be positive")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FeatureMatrixSpec":
        return cls(
            path=str(payload.get("path") or ""),
            dimension=int(payload.get("dimension") or 0),
            dtype=str(payload.get("dtype") or ""),
            checksum=str(payload.get("checksum") or ""),
            encoder=str(payload.get("encoder") or ""),
            normalized=bool(payload.get("normalized", False)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "dimension": self.dimension,
            "dtype": self.dtype,
            "checksum": self.checksum,
            "encoder": self.encoder,
            "normalized": self.normalized,
        }


@dataclass(frozen=True)
class FeatureRow:
    row_index: int
    node_id: str
    video_id: str
    lineage_hash: str

    def __post_init__(self) -> None:
        if self.row_index < 0 or not self.node_id or not self.video_id:
            raise ValueError("feature row is invalid")
        if not self.lineage_hash:
            raise ValueError("feature row requires a lineage hash")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FeatureRow":
        return cls(
            row_index=int(payload.get("row_index", -1)),
            node_id=str(payload.get("node_id") or ""),
            video_id=str(payload.get("video_id") or ""),
            lineage_hash=str(payload.get("lineage_hash") or ""),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "row_index": self.row_index,
            "node_id": self.node_id,
            "video_id": self.video_id,
            "lineage_hash": self.lineage_hash,
        }


@dataclass(frozen=True)
class FeatureStoreManifest:
    matrices: Mapping[str, FeatureMatrixSpec]
    validity: FeatureMatrixSpec
    rows: tuple[FeatureRow, ...]
    source_contract: str
    boundary_audit_version: str
    schema_version: str = FEATURE_STORE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != FEATURE_STORE_SCHEMA:
            raise ValueError(f"unsupported feature store schema: {self.schema_version}")
        if tuple(sorted(self.matrices)) != tuple(sorted(SLOT_NAMES)):
            raise ValueError(f"feature store must contain exactly {SLOT_NAMES}")
        if not self.source_contract or not self.boundary_audit_version:
            raise ValueError("feature store requires source and boundary contracts")
        indices = [row.row_index for row in self.rows]
        if indices != list(range(len(self.rows))):
            raise ValueError("feature rows must be contiguous and ordered")
        node_ids = [row.node_id for row in self.rows]
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("feature store contains duplicate node IDs")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FeatureStoreManifest":
        matrices = payload.get("matrices")
        if not isinstance(matrices, Mapping):
            raise ValueError("feature manifest matrices must be an object")
        return cls(
            schema_version=str(payload.get("schema_version") or ""),
            matrices={
                str(name): FeatureMatrixSpec.from_dict(spec)
                for name, spec in matrices.items()
                if isinstance(spec, Mapping)
            },
            validity=FeatureMatrixSpec.from_dict(payload.get("validity") or {}),
            rows=tuple(
                FeatureRow.from_dict(row)
                for row in payload.get("rows") or ()
                if isinstance(row, Mapping)
            ),
            source_contract=str(payload.get("source_contract") or ""),
            boundary_audit_version=str(payload.get("boundary_audit_version") or ""),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_contract": self.source_contract,
            "boundary_audit_version": self.boundary_audit_version,
            "matrices": {name: spec.to_dict() for name, spec in self.matrices.items()},
            "validity": self.validity.to_dict(),
            "rows": [row.to_dict() for row in self.rows],
        }

    def resolve(self, manifest_path: Path, relative_path: str) -> Path:
        path = Path(relative_path)
        return path if path.is_absolute() else manifest_path.parent / path


@dataclass(frozen=True)
class QF1CacheManifest:
    checkpoint_checksum: str
    feature_manifest_checksum: str
    matrix: FeatureMatrixSpec
    rows: tuple[FeatureRow, ...]
    num_queries: int
    hidden_size: int
    schema_version: str = QF1_CACHE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != QF1_CACHE_SCHEMA:
            raise ValueError(f"unsupported QF1 cache schema: {self.schema_version}")
        if not self.checkpoint_checksum or not self.feature_manifest_checksum:
            raise ValueError("QF1 cache requires checkpoint and feature checksums")
        if self.num_queries <= 0 or self.hidden_size <= 0:
            raise ValueError("QF1 cache dimensions must be positive")
        if self.matrix.dimension != self.num_queries * self.hidden_size:
            raise ValueError("QF1 cache matrix dimension disagrees with query shape")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "QF1CacheManifest":
        return cls(
            schema_version=str(payload.get("schema_version") or ""),
            checkpoint_checksum=str(payload.get("checkpoint_checksum") or ""),
            feature_manifest_checksum=str(payload.get("feature_manifest_checksum") or ""),
            matrix=FeatureMatrixSpec.from_dict(payload.get("matrix") or {}),
            rows=tuple(
                FeatureRow.from_dict(row)
                for row in payload.get("rows") or ()
                if isinstance(row, Mapping)
            ),
            num_queries=int(payload.get("num_queries") or 0),
            hidden_size=int(payload.get("hidden_size") or 0),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "checkpoint_checksum": self.checkpoint_checksum,
            "feature_manifest_checksum": self.feature_manifest_checksum,
            "num_queries": self.num_queries,
            "hidden_size": self.hidden_size,
            "matrix": self.matrix.to_dict(),
            "rows": [row.to_dict() for row in self.rows],
        }
