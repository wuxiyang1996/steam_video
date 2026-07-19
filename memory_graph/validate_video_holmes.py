"""End-to-end Video-Holmes validation for the Phase 1/2 memory graph."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

from .atomic_events import StructuredAtomicEventExtractor
from .contracts import L1HumanAudit
from .embedding import Qwen3VLEmbeddingProvider
from .openrouter_validation import GPTOSSGraphValidator
from .pipeline import build_causal_temporal_overlay, overlay_to_event_graph
from .schema_validation import require_valid_overlay_artifact
from .visual_verifier import QwenVisualRereadProvider
from .video_skills_l1 import VideoSkillsL1AtomicEventExtractor


DEFAULT_VIDEO_IDS = ("-3l2KFTj7yY", "0-kQB8Jo80I")


def build_parser() -> argparse.ArgumentParser:
    workspace = Path("/fs/gamma-projects/vlm-robot")
    parser = argparse.ArgumentParser(description="Validate memory graphs on Video-Holmes")
    parser.add_argument("--dataset-root", default=str(workspace / "datasets"))
    parser.add_argument("--video-skills-root", default=str(workspace / "Video_Skills"))
    parser.add_argument("--keys-py", default=str(workspace / "keys.py"))
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).parent / "outputs" / "video_holmes_validation"),
    )
    parser.add_argument("--video-id", action="append", dest="video_ids")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--memory-capacity", type=int)
    parser.add_argument(
        "--video-l1-dir",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--video-skills-l1-jsonl",
        help="Video_Skills staged examples.jsonl containing accepted clue_memory_graph L1",
    )
    parser.add_argument(
        "--input-mode",
        choices=["expert_demo", "video_only"],
        default="expert_demo",
    )
    parser.add_argument("--l1-human-audit-dir")
    parser.add_argument(
        "--allow-provisional-expert-demo",
        action="store_true",
        help="Permit causal proposals from gold segment text for structure debugging",
    )
    parser.add_argument(
        "--visual-reread",
        action="store_true",
        help="Verify causal witnesses against raw-video frames using local Qwen3.5-9B",
    )
    parser.add_argument("--visual-model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--visual-frames-per-window", type=int, default=3)
    parser.add_argument(
        "--visual-api-base",
        default="http://127.0.0.1:8000/v1/chat/completions",
    )
    parser.add_argument(
        "--require-visual-verification",
        action="store_true",
        help="Reject explains/enables whose targeted raw-video reread does not pass",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.video_l1_dir:
        raise ValueError(
            "--video-l1-dir replacement is disabled; use --video-skills-l1-jsonl"
        )
    staged_l1 = (
        _load_staged_l1_examples(Path(args.video_skills_l1_jsonl))
        if args.video_skills_l1_jsonl
        else {}
    )
    if staged_l1 and args.input_mode != "video_only":
        raise ValueError("--video-skills-l1-jsonl requires --input-mode video_only")
    video_ids = tuple(args.video_ids or staged_l1 or DEFAULT_VIDEO_IDS)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _add_video_skills_import(Path(args.video_skills_root))
    from atomic_skills.skill_model_client import SkillModelClient
    from dataset_clip_wrapper.adapters.video_holmes import VideoHolmesAdapter
    from dataset_clip_wrapper.pipeline import build_canonical_example
    from dataset_clip_wrapper.schemas import RuntimeMode, VideoRegime, WrapperConfig

    all_questions: dict[str, list[dict[str, Any]]] = {video_id: [] for video_id in video_ids}
    qa_rows = _load_qa_rows(Path(args.dataset_root), video_ids)
    selected_items = (
        {}
        if staged_l1
        else _load_selected_items(
            adapter_class=VideoHolmesAdapter,
            dataset_root=Path(args.dataset_root),
            video_ids=video_ids,
            qa_rows=qa_rows,
        )
    )
    missing = set(video_ids) - set(staged_l1 or selected_items)
    if missing:
        raise ValueError(f"Video-Holmes examples not found: {sorted(missing)}")
    for row in qa_rows:
        all_questions[str(row["video ID"])].append(row)

    runtime_mode = (
        RuntimeMode.EXPERT_DEMO
        if args.input_mode == "expert_demo"
        else RuntimeMode.VIDEO_ONLY
    )
    config = WrapperConfig(
        dataset_root=args.dataset_root,
        dataset="video_holmes",
        regime=VideoRegime.SHORT,
        mode=runtime_mode,
        split="train",
        run_backbone=False,
        run_clip_schema=False,
        run_graph_compose=False,
        run_l2_llm_planner=False,
    )
    embedder = Qwen3VLEmbeddingProvider(device=args.device)
    validator = GPTOSSGraphValidator(
        keys_py_path=args.keys_py,
        video_skills_root=args.video_skills_root,
    )
    event_extractor = (
        VideoSkillsL1AtomicEventExtractor()
        if staged_l1
        else StructuredAtomicEventExtractor(
            validator.client,
            model=validator.model,
        )
    )
    if args.require_visual_verification and not args.visual_reread:
        raise ValueError("--require-visual-verification requires --visual-reread")
    visual_provider = (
        QwenVisualRereadProvider(
            client=SkillModelClient.from_local(
                model=args.visual_model,
                base_url=args.visual_api_base,
                max_tokens=900,
                timeout_s=180,
            ),
            frames_per_window=args.visual_frames_per_window,
        )
        if args.visual_reread
        else None
    )

    summaries: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for video_id in video_ids:
        item = selected_items.get(video_id)
        canonical = (
            staged_l1[video_id]
            if staged_l1
            else build_canonical_example(item, config=config)
        )
        sample_dir = output_dir / video_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        canonical_path = sample_dir / "canonical_example.json"
        canonical_path.write_text(json.dumps(canonical, indent=2) + "\n", encoding="utf-8")
        human_audit = _load_l1_human_audit(
            Path(args.l1_human_audit_dir) if args.l1_human_audit_dir else None,
            video_id,
        )
        print(f"[{video_id}] building causal-temporal overlay...", flush=True)
        try:
            result = build_causal_temporal_overlay(
                canonical,
                input_mode=args.input_mode,
                event_extractor=event_extractor,
                human_audit=human_audit,
                embedding_provider=embedder,
                embedding_output_path=sample_dir / "event_embeddings.npy",
                relation_teacher=validator,
                batch_size=args.batch_size,
                top_k_candidates=args.top_k,
                apply_hard_verifiers=True,
                visual_reread_provider=visual_provider,
                require_visual_verification=args.require_visual_verification,
                memory_capacity=args.memory_capacity,
                allow_provisional_expert_demo=args.allow_provisional_expert_demo,
            )
            print(f"[{video_id}] overlay build complete", flush=True)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            errors.append({"video_id": video_id, "stage": "overlay_build", "error": error})
            print(f"[{video_id}] overlay build failed: {error}", flush=True)
            continue
        graph = overlay_to_event_graph(result.overlay)
        graph_path = sample_dir / "causal_temporal_overlay.json"
        overlay_payload = result.to_dict()
        try:
            require_valid_overlay_artifact(overlay_payload)
            graph_path.write_text(
                json.dumps(overlay_payload, indent=2) + "\n",
                encoding="utf-8",
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            errors.append(
                {"video_id": video_id, "stage": "schema_validation", "error": error}
            )
            print(f"[{video_id}] schema validation failed: {error}", flush=True)
            continue
        annotation = _load_annotation(Path(args.dataset_root), video_id)
        try:
            audit = validator.audit_graph(
                graph,
                held_out_annotation=annotation,
                held_out_questions=all_questions[video_id],
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            errors.append({"video_id": video_id, "stage": "graph_audit", "error": error})
            audit = {
                "error": error,
                "computed_summary": {
                    "audit_complete": False,
                    "strict_precision": None,
                    "supported_or_plausible_rate": None,
                },
            }
        audit_path = sample_dir / "audit.json"
        audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
        summaries.append(
            {
                "video_id": video_id,
                "example_id": (
                    item.example_id if item is not None else canonical.get("example_id")
                ),
                "l1_observation_count": len(result.overlay.l1_observations),
                "atomic_event_count": len(result.overlay.atomic_events),
                "deterministic_relation_count": sum(
                    relation.status.value == "deterministic"
                    for relation in graph.relations
                ),
                "candidate_relation_count": sum(
                    relation.status.value != "deterministic"
                    for relation in graph.relations
                ),
                "l1_structural_relation_count": len(
                    result.overlay.l1_structural_relations
                ),
                "l1_structural_relation_types": (
                    result.overlay.metadata.get("video_skills_l1_structural_edges")
                    or {}
                ).get("relation_counts"),
                "l1_reliability": result.l1_report.to_dict(),
                "video_skills_l1_quality": result.overlay.metadata.get(
                    "video_skills_l1_quality"
                ),
                "trust_status": result.overlay.metadata["trust_status"],
                "verifier_summary": result.verifier_summary,
                "graph_path": str(graph_path),
                "audit_path": str(audit_path),
                "audit_summary": audit.get("summary"),
                "computed_audit_summary": audit.get("computed_summary"),
                "temporal_consistency": audit.get("temporal_consistency"),
            }
        )

    summary = {
        "input_mode": args.input_mode,
        "trust_scope": (
            "provisional_structure_debug"
            if args.input_mode == "expert_demo"
            else "video_only_l1_gated"
        ),
        "embedding_model": embedder.model_name,
        "embedding_dimension": embedder.dimension,
        "relation_teacher": validator.model,
        "visual_reread_model": (
            visual_provider.model if visual_provider is not None else None
        ),
        "visual_verification_required": args.require_visual_verification,
        "video_skills_l1_jsonl": args.video_skills_l1_jsonl,
        "samples": summaries,
        "errors": errors,
        "video_count_failed": len(errors),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


def _add_video_skills_import(root: Path) -> None:
    value = str(root.resolve())
    if value not in sys.path:
        sys.path.insert(0, value)


def _load_l1_human_audit(
    audit_dir: Path | None,
    video_id: str,
) -> L1HumanAudit | None:
    if audit_dir is None:
        return None
    path = audit_dir / f"{video_id}.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain an L1 human audit object")
    return L1HumanAudit.from_dict(payload)


def _load_qa_rows(dataset_root: Path, video_ids: tuple[str, ...]) -> list[dict[str, Any]]:
    path = dataset_root / "Video-Holmes" / "Benchmark" / "train_Video-Holmes.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    selected = set(video_ids)
    return [row for row in payload if str(row.get("video ID")) in selected]


def _load_staged_l1_examples(path: Path) -> dict[str, dict[str, Any]]:
    examples: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"{path}:{line_number} must contain a JSON object")
        video = payload.get("video")
        video_id = (
            str(video.get("video_id"))
            if isinstance(video, dict) and video.get("video_id")
            else ""
        )
        if not video_id:
            raise ValueError(f"{path}:{line_number} has no video.video_id")
        if video_id in examples:
            continue
        examples[video_id] = payload
    if not examples:
        raise ValueError(f"{path} contains no staged L1 examples")
    return examples


def _load_selected_items(
    *,
    adapter_class: Any,
    dataset_root: Path,
    video_ids: tuple[str, ...],
    qa_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Run the Video_Skills adapter on an isolated subset.

    The full Video-Holmes training annotations contain unrelated malformed time
    strings (for example, a full-width semicolon). Subsetting prevents one bad
    record from blocking validation of selected examples without modifying the
    source dataset or the Video_Skills repository.
    """
    first_row_by_video: dict[str, dict[str, Any]] = {}
    for row in qa_rows:
        first_row_by_video.setdefault(str(row["video ID"]), row)

    with tempfile.TemporaryDirectory(prefix="video_holmes_subset_") as temp_dir:
        subset_root = Path(temp_dir)
        benchmark = subset_root / "Video-Holmes" / "Benchmark"
        annotation_dir = benchmark / "annotation_training"
        annotation_dir.mkdir(parents=True)
        selected_rows = [first_row_by_video[video_id] for video_id in video_ids]
        (benchmark / "train_Video-Holmes.json").write_text(
            json.dumps(selected_rows, indent=2) + "\n",
            encoding="utf-8",
        )
        source_benchmark = dataset_root / "Video-Holmes" / "Benchmark"
        for video_id in video_ids:
            source = _annotation_path(source_benchmark, video_id)
            shutil.copy2(source, annotation_dir / f"{video_id}.json")

        adapter = adapter_class(subset_root, split="train")
        items = {item.video_id: item for item in adapter.iter_items()}
        # The isolated annotation subset intentionally does not copy multi-GB
        # videos. Restore stable source paths before the temporary directory is
        # removed so raw-video VLM rereads receive real files.
        source_videos = source_benchmark / "videos_cropped"
        for video_id, item in items.items():
            source_video = source_videos / f"{video_id}.mp4"
            if source_video.exists():
                item.video_path = source_video
        return items


def _load_annotation(dataset_root: Path, video_id: str) -> dict[str, Any]:
    benchmark = dataset_root / "Video-Holmes" / "Benchmark"
    path = _annotation_path(benchmark, video_id)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list) and payload:
        return payload[0]
    if isinstance(payload, dict):
        return payload
    raise ValueError(f"annotation has an unsupported shape for {video_id}")


def _annotation_path(benchmark: Path, video_id: str) -> Path:
    for folder in ("annotations", "annotation_training"):
        path = benchmark / folder / f"{video_id}.json"
        if path.exists():
            return path
    raise FileNotFoundError(f"annotation not found for {video_id}")


if __name__ == "__main__":
    raise SystemExit(main())
