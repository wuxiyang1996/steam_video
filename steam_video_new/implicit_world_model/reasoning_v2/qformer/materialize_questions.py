"""Materialize one frozen pooled question token per retrieval case."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .feature_store import sha256_file
from .materialize_cohort import MODEL_NAME, _encode


QUESTION_PROMPT = "Represent this question for video-memory evidence retrieval."


def materialize_questions(
    labels_path: Path,
    destination: Path,
    *,
    device: str,
    batch_size: int = 16,
) -> dict[str, Any]:
    labels_path = labels_path.expanduser().resolve()
    payload = json.loads(labels_path.read_text(encoding="utf-8"))
    records = payload.get("records") or ()
    case_ids = [str(record["case_id"]) for record in records]
    if not case_ids or len(case_ids) != len(set(case_ids)):
        raise ValueError("retrieval labels require non-empty unique case IDs")
    questions = [str(record["question"]).strip() for record in records]
    if not all(questions):
        raise ValueError("retrieval labels contain an empty question")
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError("question materialization requires sentence-transformers") from exc
    model = SentenceTransformer(MODEL_NAME, device=device)
    embeddings = _encode(model, questions, QUESTION_PROMPT, batch_size).astype(np.float32)
    destination = destination.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    matrix_path = destination / "question_embeddings.npy"
    np.save(matrix_path, embeddings, allow_pickle=False)
    manifest = {
        "schema_version": "steam-qformer-question-embeddings/v0.1",
        "labels_checksum": sha256_file(labels_path),
        "encoder": f"{MODEL_NAME}:pooled-question/v1",
        "dimension": int(embeddings.shape[1]),
        "dtype": str(embeddings.dtype),
        "matrix": matrix_path.name,
        "matrix_checksum": sha256_file(matrix_path),
        "rows": [
            {"row_index": index, "case_id": case_id}
            for index, case_id in enumerate(case_ids)
        ],
        "answer_fields_present": False,
    }
    manifest_path = destination / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {"manifest": str(manifest_path), "shape": list(embeddings.shape)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args(argv)
    report = materialize_questions(
        args.labels, args.destination, device=args.device, batch_size=args.batch_size
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
