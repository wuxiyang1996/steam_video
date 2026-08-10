"""Validated read-only access to frozen QF1 node tokens."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .contracts import QF1CacheManifest
from .feature_store import sha256_file


class QF1Cache:
    def __init__(
        self,
        manifest_path: str | Path,
        *,
        feature_manifest_path: str | Path | None = None,
        verify_checksums: bool = True,
        mmap_mode: str | None = "r",
    ) -> None:
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.manifest = QF1CacheManifest.from_dict(payload)
        matrix_path = Path(self.manifest.matrix.path)
        if not matrix_path.is_absolute():
            matrix_path = self.manifest_path.parent / matrix_path
        if verify_checksums and sha256_file(matrix_path) != self.manifest.matrix.checksum:
            raise ValueError(f"QF1 cache checksum mismatch: {matrix_path}")
        if feature_manifest_path is not None:
            feature_path = Path(feature_manifest_path).expanduser().resolve()
            if sha256_file(feature_path) != self.manifest.feature_manifest_checksum:
                raise ValueError("QF1 cache feature-manifest lineage mismatch")
        self.tokens = np.load(matrix_path, mmap_mode=mmap_mode, allow_pickle=False)
        expected = (
            len(self.manifest.rows),
            self.manifest.num_queries,
            self.manifest.hidden_size,
        )
        if self.tokens.shape != expected:
            raise ValueError(f"QF1 cache has shape {self.tokens.shape}; expected {expected}")
        if str(self.tokens.dtype) != self.manifest.matrix.dtype:
            raise ValueError("QF1 cache dtype disagrees with manifest")
        if not np.isfinite(self.tokens).all():
            raise ValueError("QF1 cache contains non-finite values")
        self._index = {row.node_id: row.row_index for row in self.manifest.rows}

    def __len__(self) -> int:
        return len(self.manifest.rows)

    def __contains__(self, node_id: str) -> bool:
        return node_id in self._index

    def node(self, node_id: str) -> np.ndarray:
        try:
            return self.tokens[self._index[node_id]]
        except KeyError as exc:
            raise KeyError(f"unknown QF1 node: {node_id}") from exc
