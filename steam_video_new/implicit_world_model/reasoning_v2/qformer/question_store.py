"""Validated read-only frozen pooled question embeddings."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .feature_store import sha256_file


class QuestionEmbeddingStore:
    def __init__(self, manifest_path: str | Path, *, labels_path: str | Path) -> None:
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != "steam-qformer-question-embeddings/v0.1":
            raise ValueError("unsupported question embedding schema")
        if payload.get("answer_fields_present") is not False:
            raise ValueError("question embedding manifest failed answer leakage gate")
        labels_path = Path(labels_path).expanduser().resolve()
        if sha256_file(labels_path) != payload.get("labels_checksum"):
            raise ValueError("question embeddings disagree with retrieval labels")
        matrix_path = self.manifest_path.parent / str(payload["matrix"])
        if sha256_file(matrix_path) != payload.get("matrix_checksum"):
            raise ValueError("question embedding checksum mismatch")
        self.embeddings = np.load(matrix_path, mmap_mode="r", allow_pickle=False)
        expected = (len(payload["rows"]), int(payload["dimension"]))
        if self.embeddings.shape != expected or str(self.embeddings.dtype) != payload["dtype"]:
            raise ValueError("question embedding shape or dtype mismatch")
        self._index = {
            str(row["case_id"]): int(row["row_index"]) for row in payload["rows"]
        }

    @property
    def dimension(self) -> int:
        return int(self.embeddings.shape[1])

    def case(self, case_id: str) -> np.ndarray:
        try:
            return self.embeddings[self._index[case_id]]
        except KeyError as exc:
            raise KeyError(f"unknown question case: {case_id}") from exc
