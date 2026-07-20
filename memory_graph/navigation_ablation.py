"""Compare four bounded graph-navigation strategies on fixed evidence targets."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import deque
from pathlib import Path
from typing import Any, Sequence


STRATEGIES = (
    "semantic_only",
    "event_only",
    "native_l1_candidate",
    "verified_dependency",
)
TEMPORAL_RELATIONS = frozenset({"temporal_next", "before", "overlaps", "during"})
DEPENDENCY_RELATIONS = frozenset({"transition_support", "state_transition"})
CANDIDATE_L1_RELATIONS = frozenset(
    {
        "same_instance_candidate",
        "reappears_candidate",
        "observation_support",
        "same_entity",
        "same_object",
    }
)


def evaluate_navigation_ablation(
    overlay: dict[str, Any],
    cases: list[dict[str, Any]],
    *,
    event_embeddings: dict[str, Sequence[float]] | None = None,
    query_embeddings: dict[str, Sequence[float]] | None = None,
    embedding_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    events = {
        str(node["node_id"]): node
        for node in overlay.get("atomic_events") or []
        if isinstance(node, dict) and node.get("node_id")
    }
    results: dict[str, list[dict[str, Any]]] = {name: [] for name in STRATEGIES}
    used_embedding_retrieval = False
    for case in cases:
        case_id = str(case.get("case_id") or "")
        question = str(case.get("question") or "")
        gold = {str(value) for value in case.get("gold_event_ids") or []}
        unknown = gold - set(events)
        if not question or not gold or unknown:
            raise ValueError(
                f"case {case.get('case_id')} requires a question and known gold_event_ids; "
                f"unknown={sorted(unknown)}"
            )
        budget = int(case.get("graph_read_budget") or 4)
        if budget < 1:
            raise ValueError("graph_read_budget must be positive")
        scores, retrieval_mode = _relevance_scores(
            case_id=case_id,
            question=question,
            events=events,
            event_embeddings=event_embeddings,
            query_embeddings=query_embeddings,
        )
        used_embedding_retrieval |= retrieval_mode == "qwen3_vl_embedding"
        for strategy in STRATEGIES:
            acquired = _run_strategy(
                strategy,
                relevance_scores=scores,
                budget=budget,
                events=events,
                overlay=overlay,
            )
            hits = gold & set(acquired)
            results[strategy].append(
                {
                    "case_id": case.get("case_id"),
                    "acquired_event_ids": acquired,
                    "graph_reads": len(acquired),
                    "gold_count": len(gold),
                    "hit_count": len(hits),
                    "evidence_recall": len(hits) / len(gold),
                    "answerable": hits == gold,
                    "retrieval_mode": retrieval_mode,
                    "query_embedding_ref": case.get("query_embedding_ref"),
                }
            )
    summary = {
        strategy: _summarize(rows) for strategy, rows in results.items()
    }
    return {
        "schema_version": "steam-navigation-ablation/v0.1",
        "case_count": len(cases),
        "strategies": summary,
        "cases": results,
        "semantics": {
            "answerable": "all independently specified gold evidence events acquired",
            "graph_reads": "persisted event reads; imagined observations are excluded",
            "native_l1_candidate": "candidate/native L1 edges may choose reads but are not answer evidence",
            "verified_dependency": "only hard-verified or accepted-track-derived dependency edges",
        },
        "retrieval": {
            "mode": (
                "qwen3_vl_embedding"
                if used_embedding_retrieval
                else "lexical_fallback"
            ),
            **dict(embedding_metadata or {}),
        },
    }


def _run_strategy(
    strategy: str,
    *,
    relevance_scores: dict[str, float],
    budget: int,
    events: dict[str, dict[str, Any]],
    overlay: dict[str, Any],
) -> list[str]:
    ranked = sorted(
        events,
        key=lambda event_id: (
            -float(relevance_scores.get(event_id, 0.0)),
            float((events[event_id].get("time_span") or {}).get("start_s") or 0.0),
            event_id,
        ),
    )
    if strategy == "semantic_only":
        return ranked[:budget]
    acquired = [ranked[0]] if ranked else []
    adjacency = _event_adjacency(overlay, verified_only=strategy == "verified_dependency")
    if strategy == "native_l1_candidate":
        _add_native_l1_adjacency(adjacency, overlay, events)
    queue = deque(acquired)
    while queue and len(acquired) < budget:
        current = queue.popleft()
        neighbors = sorted(
            adjacency.get(current, ()),
            key=lambda event_id: (
                -float(relevance_scores.get(event_id, 0.0)),
                event_id,
            ),
        )
        for neighbor in neighbors:
            if neighbor in acquired or neighbor not in events:
                continue
            acquired.append(neighbor)
            queue.append(neighbor)
            if len(acquired) >= budget:
                break
    for event_id in ranked:
        if len(acquired) >= budget:
            break
        if event_id not in acquired:
            acquired.append(event_id)
    return acquired


def _event_adjacency(
    overlay: dict[str, Any],
    *,
    verified_only: bool,
) -> dict[str, set[str]]:
    adjacency: dict[str, set[str]] = {}
    for edge in overlay.get("relations") or []:
        if not isinstance(edge, dict):
            continue
        names = set((edge.get("relation_probabilities") or {}).keys())
        allowed = bool(names & TEMPORAL_RELATIONS)
        if verified_only and names & DEPENDENCY_RELATIONS:
            allowed = allowed or _dependency_verified(edge)
        if not verified_only:
            allowed = allowed and not bool(names & DEPENDENCY_RELATIONS)
        if not allowed:
            continue
        src, dst = str(edge.get("src") or ""), str(edge.get("dst") or "")
        adjacency.setdefault(src, set()).add(dst)
        adjacency.setdefault(dst, set()).add(src)
    return adjacency


def _dependency_verified(edge: dict[str, Any]) -> bool:
    provenance = edge.get("provenance") or {}
    if provenance.get("accepted_identity_track"):
        return True
    verifier = provenance.get("hard_verifier")
    return isinstance(verifier, dict) and any(
        isinstance(value, dict) and value.get("passed") is True
        for name, value in verifier.items()
        if name in DEPENDENCY_RELATIONS
    )


def _add_native_l1_adjacency(
    adjacency: dict[str, set[str]],
    overlay: dict[str, Any],
    events: dict[str, dict[str, Any]],
) -> None:
    event_by_l1: dict[str, set[str]] = {}
    l1_by_id = {
        str(node.get("node_id")): node
        for node in overlay.get("l1_observations") or []
        if isinstance(node, dict) and node.get("node_id")
    }
    events_by_clip: dict[str, set[str]] = {}
    for event_id, event in events.items():
        for ref in event.get("source_segments") or []:
            l1_ref = str(ref)
            event_by_l1.setdefault(l1_ref, set()).add(event_id)
            clip_id = str(
                ((l1_by_id.get(l1_ref) or {}).get("metadata") or {}).get(
                    "clip_id"
                )
                or ""
            )
            if clip_id:
                events_by_clip.setdefault(clip_id, set()).add(event_id)
    for edge in overlay.get("l1_structural_relations") or []:
        if not isinstance(edge, dict):
            continue
        names = set((edge.get("relation_probabilities") or {}).keys())
        if not names & CANDIDATE_L1_RELATIONS:
            continue
        src_id, dst_id = str(edge.get("src") or ""), str(edge.get("dst") or "")
        src_clip = str(
            ((l1_by_id.get(src_id) or {}).get("metadata") or {}).get("clip_id")
            or ""
        )
        dst_clip = str(
            ((l1_by_id.get(dst_id) or {}).get("metadata") or {}).get("clip_id")
            or ""
        )
        src_events = event_by_l1.get(src_id, set()) or events_by_clip.get(
            src_clip, set()
        )
        dst_events = event_by_l1.get(dst_id, set()) or events_by_clip.get(
            dst_clip, set()
        )
        for src in src_events:
            for dst in dst_events:
                if src == dst:
                    continue
                adjacency.setdefault(src, set()).add(dst)
                adjacency.setdefault(dst, set()).add(src)


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    return {
        "case_count": count,
        "answer_accuracy": sum(bool(row["answerable"]) for row in rows) / count
        if count
        else None,
        "mean_evidence_recall": sum(float(row["evidence_recall"]) for row in rows)
        / count
        if count
        else None,
        "mean_graph_reads": sum(int(row["graph_reads"]) for row in rows) / count
        if count
        else None,
    }


def _event_text(event: dict[str, Any]) -> str:
    return str((event.get("metadata") or {}).get("predicate") or event.get("text") or "")


def _lexical_score(question: str, text: str) -> int:
    return len(_tokens(question) & _tokens(text))


def _tokens(value: str) -> set[str]:
    stop = {
        "a",
        "an",
        "and",
        "between",
        "did",
        "how",
        "is",
        "on",
        "the",
        "to",
        "was",
        "what",
        "who",
    }
    aliases = {"male": "man", "males": "man", "men": "man"}
    return {
        aliases.get(token, token)
        for token in re.findall(r"[a-z0-9]+", value.casefold())
        if len(token) > 1 and token not in stop
    }


def _relevance_scores(
    *,
    case_id: str,
    question: str,
    events: dict[str, dict[str, Any]],
    event_embeddings: dict[str, Sequence[float]] | None,
    query_embeddings: dict[str, Sequence[float]] | None,
) -> tuple[dict[str, float], str]:
    query = (query_embeddings or {}).get(case_id)
    if query is not None and event_embeddings is not None:
        query_values = [float(value) for value in query]
        scores: dict[str, float] = {}
        for event_id in events:
            vector = event_embeddings.get(event_id)
            if vector is None or len(vector) != len(query_values):
                raise ValueError(
                    f"embedding dimension mismatch for event {event_id}"
                )
            scores[event_id] = sum(
                left * float(right)
                for left, right in zip(query_values, vector)
            )
        return scores, "qwen3_vl_embedding"
    return (
        {
            event_id: float(_lexical_score(question, _event_text(event)))
            for event_id, event in events.items()
        },
        "lexical_fallback",
    )


def prepare_qwen_navigation_embeddings(
    overlay: dict[str, Any],
    cases: list[dict[str, Any]],
    *,
    output_path: Path,
    device: str | None = None,
    provider: Any | None = None,
) -> tuple[
    list[dict[str, Any]],
    dict[str, Sequence[float]],
    dict[str, Sequence[float]],
    dict[str, Any],
]:
    """Load persisted event vectors and embed questions with the same Qwen model."""
    import numpy as np

    from .embedding import Qwen3VLEmbeddingProvider
    from .types import DEFAULT_EMBEDDING_DIM, DEFAULT_EMBEDDING_MODEL

    encoder = provider or Qwen3VLEmbeddingProvider(device=device)
    if encoder.model_name != DEFAULT_EMBEDDING_MODEL:
        raise ValueError(
            f"navigation embeddings require {DEFAULT_EMBEDDING_MODEL}, "
            f"got {encoder.model_name}"
        )
    if int(encoder.dimension) != DEFAULT_EMBEDDING_DIM:
        raise ValueError(
            f"navigation embedding dimension must be {DEFAULT_EMBEDDING_DIM}"
        )

    matrix_cache: dict[str, Any] = {}
    checksum_cache: dict[str, str] = {}
    event_embeddings: dict[str, Sequence[float]] = {}
    for event in overlay.get("atomic_events") or []:
        if not isinstance(event, dict) or not event.get("node_id"):
            continue
        ref = event.get("embedding_ref") or {}
        if ref.get("model") != DEFAULT_EMBEDDING_MODEL:
            raise ValueError(
                f"event {event['node_id']} lacks a {DEFAULT_EMBEDDING_MODEL} embedding"
            )
        if int(ref.get("dimension") or 0) != DEFAULT_EMBEDDING_DIM:
            raise ValueError(f"event {event['node_id']} has an invalid embedding dimension")
        path = str(ref.get("path") or "")
        row_index = int(ref.get("row_index", -1))
        if not path or row_index < 0:
            raise ValueError(f"event {event['node_id']} has an incomplete embedding_ref")
        if path not in matrix_cache:
            matrix_path = Path(path)
            payload = matrix_path.read_bytes()
            actual_checksum = hashlib.sha256(payload).hexdigest()
            expected_checksum = str(ref.get("checksum") or "")
            if expected_checksum and actual_checksum != expected_checksum:
                raise ValueError(f"embedding checksum mismatch for {matrix_path}")
            matrix_cache[path] = np.load(matrix_path)
            checksum_cache[path] = actual_checksum
        matrix = matrix_cache[path]
        if row_index >= len(matrix):
            raise ValueError(f"embedding row {row_index} is out of range for {path}")
        event_embeddings[str(event["node_id"])] = matrix[row_index]

    questions = [str(case.get("question") or "") for case in cases]
    query_matrix = np.asarray(encoder.encode(questions, batch_size=8), dtype=np.float32)
    expected_shape = (len(cases), DEFAULT_EMBEDDING_DIM)
    if query_matrix.shape != expected_shape:
        raise ValueError(
            f"query embedding shape {query_matrix.shape}; expected {expected_shape}"
        )
    norms = np.linalg.norm(query_matrix, axis=1, keepdims=True)
    query_matrix = query_matrix / np.clip(norms, 1e-12, None)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, query_matrix)
    saved_path = (
        output_path if output_path.suffix == ".npy" else output_path.with_suffix(".npy")
    )
    checksum = hashlib.sha256(saved_path.read_bytes()).hexdigest()
    updated_cases: list[dict[str, Any]] = []
    query_embeddings: dict[str, Sequence[float]] = {}
    for row_index, (case, vector) in enumerate(zip(cases, query_matrix)):
        case_id = str(case.get("case_id") or "")
        updated = dict(case)
        updated["query_embedding_ref"] = {
            "path": str(saved_path),
            "model": DEFAULT_EMBEDDING_MODEL,
            "dimension": DEFAULT_EMBEDDING_DIM,
            "dtype": "float32",
            "normalized": True,
            "row_index": row_index,
            "checksum": checksum,
        }
        updated_cases.append(updated)
        query_embeddings[case_id] = vector
    manifest_path = saved_path.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(
            {
                "model": DEFAULT_EMBEDDING_MODEL,
                "dimension": DEFAULT_EMBEDDING_DIM,
                "normalized": True,
                "matrix_path": str(saved_path),
                "checksum": checksum,
                "rows": [
                    {
                        "row_index": index,
                        "case_id": str(case.get("case_id") or ""),
                    }
                    for index, case in enumerate(cases)
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return (
        updated_cases,
        event_embeddings,
        query_embeddings,
        {
            "model": DEFAULT_EMBEDDING_MODEL,
            "dimension": DEFAULT_EMBEDDING_DIM,
            "normalized": True,
            "query_matrix_path": str(saved_path),
            "query_matrix_checksum": checksum,
            "event_matrix_paths": sorted(matrix_cache),
            "event_matrix_checksums": checksum_cache,
        },
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overlay", required=True, type=Path)
    parser.add_argument("--cases", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--query-embedding-output", type=Path)
    parser.add_argument("--embedding-device")
    args = parser.parse_args(argv)
    overlay = json.loads(args.overlay.read_text(encoding="utf-8"))
    cases_payload = json.loads(args.cases.read_text(encoding="utf-8"))
    cases = cases_payload.get("cases") if isinstance(cases_payload, dict) else cases_payload
    if not isinstance(cases, list):
        raise ValueError("cases JSON must be an array or an object with a cases array")
    event_embeddings = None
    query_embeddings = None
    embedding_metadata = None
    if args.query_embedding_output:
        (
            cases,
            event_embeddings,
            query_embeddings,
            embedding_metadata,
        ) = prepare_qwen_navigation_embeddings(
            overlay,
            cases,
            output_path=args.query_embedding_output,
            device=args.embedding_device,
        )
    report = evaluate_navigation_ablation(
        overlay,
        cases,
        event_embeddings=event_embeddings,
        query_embeddings=query_embeddings,
        embedding_metadata=embedding_metadata,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["strategies"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
