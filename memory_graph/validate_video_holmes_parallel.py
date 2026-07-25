"""Parallel Video-Holmes validation: GPU embedding overlapped with OpenRouter API."""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from .adapter import canonical_to_memory_nodes
from .embedding import Qwen3VLEmbeddingProvider, embed_memory_nodes
from .graph_builder import build_memory_graph
from .validate_video_holmes import (
    _load_qa_rows,
    _load_selected_items,
    main as trusted_pipeline_main,
)


def build_parser() -> argparse.ArgumentParser:
    workspace = Path("/fs/gamma-projects/vlm-robot")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default=str(workspace / "datasets"))
    parser.add_argument("--video-skills-root", default=str(workspace / "Video_Skills"))
    parser.add_argument("--keys-py", default=str(workspace / "keys.py"))
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).parent / "outputs" / "video_holmes_50"),
    )
    parser.add_argument(
        "--video-ids-json",
        default=str(
            Path(__file__).parent / "outputs" / "video_holmes_50" / "video_ids.json"
        ),
        help="JSON with a video_ids list; if missing, the first --limit usable IDs are chosen",
    )
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument(
        "--api-workers",
        type=int,
        default=8,
        help="Concurrent OpenRouter label/audit workers",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Reuse samples that already have audit.json",
    )
    parser.add_argument(
        "--input-mode",
        choices=["expert_demo", "video_only"],
        default="expert_demo",
    )
    parser.add_argument("--l1-human-audit-dir")
    parser.add_argument("--allow-provisional-expert-demo", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _add_video_skills_import(Path(args.video_skills_root))
    from dataset_clip_wrapper.adapters.video_holmes import VideoHolmesAdapter

    video_ids = _resolve_video_ids(
        video_ids_json=Path(args.video_ids_json),
        dataset_root=Path(args.dataset_root),
        limit=args.limit,
        adapter_class=VideoHolmesAdapter,
    )
    delegated = [
        "--dataset-root",
        args.dataset_root,
        "--video-skills-root",
        args.video_skills_root,
        "--keys-py",
        args.keys_py,
        "--output-dir",
        args.output_dir,
        "--device",
        args.device,
        "--batch-size",
        str(args.batch_size),
        "--top-k",
        str(args.top_k),
        "--input-mode",
        args.input_mode,
    ]
    for video_id in video_ids:
        delegated.append(f"--video-id={video_id}")
    if args.l1_human_audit_dir:
        delegated.extend(["--l1-human-audit-dir", args.l1_human_audit_dir])
    if args.allow_provisional_expert_demo:
        delegated.append("--allow-provisional-expert-demo")
    print(
        "[notice] API-only legacy parallel path is disabled; running the unified "
        "atomic-event pipeline sequentially.",
        flush=True,
    )
    return trusted_pipeline_main(delegated)

    # Legacy implementation retained below for reading historical outputs only.
    started = time.time()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _add_video_skills_import(Path(args.video_skills_root))
    from dataset_clip_wrapper.adapters.video_holmes import VideoHolmesAdapter
    from dataset_clip_wrapper.schemas import RuntimeMode, VideoRegime, WrapperConfig

    video_ids = _resolve_video_ids(
        video_ids_json=Path(args.video_ids_json),
        dataset_root=Path(args.dataset_root),
        limit=args.limit,
        adapter_class=VideoHolmesAdapter,
    )
    (output_dir / "selected_video_ids.json").write_text(
        json.dumps({"video_ids": list(video_ids)}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"[setup] validating {len(video_ids)} videos with {args.api_workers} API workers",
        flush=True,
    )

    qa_rows = _load_qa_rows(Path(args.dataset_root), video_ids)
    selected_items = _load_selected_items(
        adapter_class=VideoHolmesAdapter,
        dataset_root=Path(args.dataset_root),
        video_ids=video_ids,
        qa_rows=qa_rows,
    )
    missing = set(video_ids) - set(selected_items)
    if missing:
        raise ValueError(f"Video-Holmes examples not found: {sorted(missing)}")

    all_questions: dict[str, list[dict[str, Any]]] = {
        video_id: [] for video_id in video_ids
    }
    for row in qa_rows:
        all_questions[str(row["video ID"])].append(row)

    config = WrapperConfig(
        dataset_root=args.dataset_root,
        dataset="video_holmes",
        regime=VideoRegime.SHORT,
        mode=RuntimeMode.EXPERT_DEMO,
        split="train",
        run_backbone=False,
        run_clip_schema=False,
        run_graph_compose=False,
        run_l2_llm_planner=False,
    )
    embedder = Qwen3VLEmbeddingProvider(device=args.device)
    print(
        f"[setup] embedding model={embedder.model_name} dim={embedder.dimension} device={args.device}",
        flush=True,
    )

    summaries: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    # Process pool: OpenRouter client's SIGALRM timeout is process-local and
    # does not work reliably inside ThreadPoolExecutor workers.
    futures: dict[Future[dict[str, Any]], str] = {}

    with ProcessPoolExecutor(max_workers=args.api_workers) as pool:
        for index, video_id in enumerate(video_ids, start=1):
            sample_dir = output_dir / video_id
            audit_path = sample_dir / "audit.json"
            if args.skip_existing and audit_path.exists():
                print(
                    f"[{index}/{len(video_ids)}] skip existing {video_id}", flush=True
                )
                summaries.append(_summary_from_existing(sample_dir, video_id))
                continue

            try:
                prepared = _prepare_sample(
                    video_id=video_id,
                    item=selected_items[video_id],
                    config=config,
                    embedder=embedder,
                    output_dir=output_dir,
                    batch_size=args.batch_size,
                    top_k=args.top_k,
                )
            except Exception as exc:
                err = {
                    "video_id": video_id,
                    "stage": "prepare_embed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
                errors.append(err)
                print(
                    f"[{index}/{len(video_ids)}] PREPARE FAIL {video_id}: {err['error']}",
                    flush=True,
                )
                continue

            print(
                f"[{index}/{len(video_ids)}] embedded {video_id} "
                f"nodes={prepared['node_count']} -> API queue",
                flush=True,
            )
            _persist_prepared_for_workers(prepared)
            job = {
                "video_id": video_id,
                "sample_dir": prepared["sample_dir"],
                "dataset_root": str(args.dataset_root),
                "keys_py": args.keys_py,
                "video_skills_root": args.video_skills_root,
                "top_k": args.top_k,
            }
            futures[pool.submit(_label_and_audit_job, job)] = video_id

            # Drain finished API jobs so GPU embedding and OpenRouter overlap.
            done = [future for future in futures if future.done()]
            for future in done:
                _consume_api_future(future, futures, summaries, errors)

        for future in as_completed(list(futures)):
            _consume_api_future(future, futures, summaries, errors)

    summaries.sort(key=lambda row: str(row.get("video_id") or ""))
    evaluation = evaluate_summaries(summaries)
    summary = {
        "embedding_model": embedder.model_name,
        "embedding_dimension": embedder.dimension,
        "relation_teacher": "openai/gpt-oss-120b",
        "video_count_requested": len(video_ids),
        "video_count_succeeded": len(summaries),
        "video_count_failed": len(errors),
        "api_workers": args.api_workers,
        "elapsed_s": round(time.time() - started, 2),
        "evaluation": evaluation,
        "samples": summaries,
        "errors": errors,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "evaluation.json").write_text(
        json.dumps(evaluation, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {"evaluation": evaluation, "elapsed_s": summary["elapsed_s"]}, indent=2
        ),
        flush=True,
    )
    return 0 if not errors else 2


def evaluate_summaries(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    supported = 0
    plausible = 0
    unsupported = 0
    audited = 0
    temporal_pass = 0
    temporal_total = 0
    complete_audits = 0
    per_video: list[dict[str, Any]] = []

    for row in summaries:
        computed = row.get("computed_audit_summary") or {}
        if not computed:
            continue
        s = int(computed.get("supported") or 0)
        p = int(computed.get("plausible") or 0)
        u = int(computed.get("unsupported_or_contradicted") or 0)
        a = int(computed.get("audited_predictions") or 0)
        supported += s
        plausible += p
        unsupported += u
        audited += a
        if computed.get("audit_complete"):
            complete_audits += 1
        temporal = row.get("temporal_consistency") or {}
        if "passed" in temporal:
            temporal_total += 1
            if temporal.get("passed"):
                temporal_pass += 1
        per_video.append(
            {
                "video_id": row.get("video_id"),
                "strict_precision": computed.get("strict_precision"),
                "supported_or_plausible_rate": computed.get(
                    "supported_or_plausible_rate"
                ),
                "audited_predictions": a,
                "temporal_passed": temporal.get("passed"),
            }
        )

    strict = supported / audited if audited else 0.0
    soft = (supported + plausible) / audited if audited else 0.0
    gate_strict = 0.70
    return {
        "videos_evaluated": len(summaries),
        "complete_audits": complete_audits,
        "audited_predictions": audited,
        "supported": supported,
        "plausible": plausible,
        "unsupported_or_contradicted": unsupported,
        "strict_precision": strict,
        "supported_or_plausible_rate": soft,
        "temporal_pass_rate": (temporal_pass / temporal_total)
        if temporal_total
        else None,
        "gate_strict_precision": gate_strict,
        "candidate_causal_gate": "pass" if strict >= gate_strict else "fail",
        "temporal_gate": (
            "pass"
            if temporal_total and temporal_pass == temporal_total
            else ("fail" if temporal_total else "incomplete")
        ),
        "per_video": per_video,
    }


def _prepare_sample(
    *,
    video_id: str,
    item: Any,
    config: Any,
    embedder: Qwen3VLEmbeddingProvider,
    output_dir: Path,
    batch_size: int,
    top_k: int,
) -> dict[str, Any]:
    from dataset_clip_wrapper.pipeline import build_canonical_example

    canonical = build_canonical_example(item, config=config)
    source_graph, all_nodes = canonical_to_memory_nodes(canonical)
    nodes = [
        node
        for node in all_nodes
        if node.metadata.get("source_type") == "segment_description"
    ]
    if len(nodes) < 2:
        raise ValueError(
            f"{video_id} has fewer than two timestamped segment-description nodes"
        )

    sample_dir = output_dir / video_id
    sample_dir.mkdir(parents=True, exist_ok=True)
    (sample_dir / "canonical_example.json").write_text(
        json.dumps(canonical, indent=2) + "\n",
        encoding="utf-8",
    )
    embeddings = embed_memory_nodes(
        nodes,
        embedder,
        output_path=sample_dir / "node_embeddings.npy",
        batch_size=batch_size,
    )
    graph = build_memory_graph(
        graph_id=f"memory_graph:{item.example_id}",
        example_id=item.example_id,
        video_id=video_id,
        nodes=nodes,
        top_k_candidates=top_k,
    )
    return {
        "video_id": video_id,
        "example_id": item.example_id,
        "sample_dir": str(sample_dir),
        "nodes": nodes,
        "embeddings": embeddings,
        "graph": graph,
        "source_graph": source_graph,
        "node_count": len(nodes),
        "deterministic_relation_count": len(graph.relations),
    }


def _persist_prepared_for_workers(prepared: dict[str, Any]) -> None:
    sample_dir = Path(prepared["sample_dir"])
    # Canonical JSON and embeddings are already written by _prepare_sample.
    # Keep a tiny marker so retries know GPU prep finished.
    (sample_dir / "prepared.ok").write_text("ok\n", encoding="utf-8")


def _consume_api_future(
    future: Future[dict[str, Any]],
    futures: dict[Future[dict[str, Any]], str],
    summaries: list[dict[str, Any]],
    errors: list[dict[str, Any]],
) -> None:
    video_id = futures.pop(future, "unknown")
    try:
        result = future.result()
    except Exception as exc:
        err = {
            "video_id": video_id,
            "stage": "api",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
        errors.append(err)
        print(f"[api] FAIL {video_id}: {err['error']}", flush=True)
        return
    if result.get("error"):
        errors.append(result)
        print(f"[api] FAIL {result['video_id']}: {result['error']}", flush=True)
        return
    summaries.append(result)
    computed = result.get("computed_audit_summary") or {}
    print(
        f"[api] ok {result['video_id']} "
        f"strict_precision={computed.get('strict_precision')} "
        f"supported={computed.get('supported')}/"
        f"{computed.get('audited_predictions')}",
        flush=True,
    )


def _label_and_audit_job(job: dict[str, Any]) -> dict[str, Any]:
    """Process-pool entrypoint: reload prepared sample from disk and call OpenRouter."""
    from .retry_failed_api import _label_and_audit_from_disk

    video_id = job["video_id"]
    try:
        result = _label_and_audit_from_disk(
            video_id=video_id,
            sample_dir=Path(job["sample_dir"]),
            dataset_root=Path(job["dataset_root"]),
            keys_py=job["keys_py"],
            video_skills_root=job["video_skills_root"],
            top_k=int(job["top_k"]),
            timeout_s=180,
        )
        return result
    except Exception as exc:
        return {
            "video_id": video_id,
            "stage": "api",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }


def _summary_from_existing(sample_dir: Path, video_id: str) -> dict[str, Any]:
    audit = json.loads((sample_dir / "audit.json").read_text(encoding="utf-8"))
    graph = json.loads((sample_dir / "memory_graph.json").read_text(encoding="utf-8"))
    deterministic = sum(
        1
        for relation in graph.get("relations") or []
        if relation.get("status") == "deterministic"
    )
    candidate = len(graph.get("relations") or []) - deterministic
    return {
        "video_id": video_id,
        "example_id": graph.get("example_id"),
        "node_count": len(graph.get("nodes") or []),
        "deterministic_relation_count": deterministic,
        "candidate_relation_count": candidate,
        "graph_path": str(sample_dir / "memory_graph.json"),
        "audit_path": str(sample_dir / "audit.json"),
        "audit_summary": audit.get("summary"),
        "computed_audit_summary": audit.get("computed_summary"),
        "temporal_consistency": audit.get("temporal_consistency"),
        "reused_existing": True,
    }


def _resolve_video_ids(
    *,
    video_ids_json: Path,
    dataset_root: Path,
    limit: int,
    adapter_class: Any,
) -> tuple[str, ...]:
    if video_ids_json.exists():
        payload = json.loads(video_ids_json.read_text(encoding="utf-8"))
        ids = [str(value) for value in payload.get("video_ids") or []]
        if not ids:
            raise ValueError(f"{video_ids_json} contains no video_ids")
        return tuple(ids[:limit])

    qa = json.loads(
        (
            dataset_root / "Video-Holmes" / "Benchmark" / "train_Video-Holmes.json"
        ).read_text(encoding="utf-8")
    )
    by_vid: dict[str, dict[str, Any]] = {}
    for row in qa:
        by_vid.setdefault(str(row["video ID"]), row)
    ann_dir = dataset_root / "Video-Holmes" / "Benchmark" / "annotation_training"
    selected: list[str] = []
    for video_id in sorted(by_vid):
        path = ann_dir / f"{video_id}.json"
        if not path.exists() or "；" in path.read_text(encoding="utf-8"):
            continue
        selected.append(video_id)
        if len(selected) >= limit:
            break
    if len(selected) < limit:
        raise ValueError(f"only found {len(selected)} candidate videos; need {limit}")
    return tuple(selected)


def _add_video_skills_import(root: Path) -> None:
    value = str(root.resolve())
    if value not in sys.path:
        sys.path.insert(0, value)


if __name__ == "__main__":
    raise SystemExit(main())
