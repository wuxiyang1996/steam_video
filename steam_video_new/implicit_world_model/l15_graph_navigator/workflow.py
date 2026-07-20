"""Dataset, annotation, training-export, and matched-ablation workflow CLI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any

from .case_miner import mine_navigation_cases
from .executed_transitions import (
    build_executed_transition_dataset,
    export_executed_transition_training_records,
    lock_executed_transition_dataset,
    validate_executed_transition_dataset,
)
from .matched_ablation import evaluate_matched_navigation
from .overlay_io import load_overlay_artifact
from .preference_data import (
    build_preference_annotation_packet,
    export_training_records,
    lock_annotation_packet,
    lock_navigation_case_set,
    require_valid_navigation_case_set,
    validate_navigation_case_set,
)
from .run import main as run_one
from .train_models import train_baselines
from .video_skills_adapter import VideoSkillsL2Adapter


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    mine = commands.add_parser("mine-cases")
    mine.add_argument("--overlay-root", type=Path)
    mine.add_argument(
        "--overlay",
        action="append",
        default=[],
        type=Path,
        help="Additional explicit overlay artifact; may be repeated.",
    )
    mine.add_argument("--glob", default="**/causal_temporal_overlay.json")
    mine.add_argument("--case-set-id", required=True)
    mine.add_argument("--desired-count", type=int, default=40)
    mine.add_argument("--per-video-limit", type=int, default=5)
    mine.add_argument("--output", required=True, type=Path)
    mine.add_argument("--report", required=True, type=Path)

    validate = commands.add_parser("validate-cases")
    validate.add_argument("--cases", required=True, type=Path)

    lock_cases = commands.add_parser("lock-cases")
    lock_cases.add_argument("--cases", required=True, type=Path)
    lock_cases.add_argument("--output", required=True, type=Path)
    lock_cases.add_argument("--annotator", required=True)
    lock_cases.add_argument(
        "--status", choices=("ai_provisional", "human_locked"), required=True
    )

    generate = commands.add_parser("generate")
    generate.add_argument("--cases", required=True, type=Path)
    generate.add_argument("--output-dir", required=True, type=Path)
    generate.add_argument("--video-skills-root", type=Path)
    generate.add_argument("--no-video-skills-runtime", action="store_true")
    generate.add_argument(
        "--belief-backend", choices=("factor_graph", "factorized"), default="factor_graph"
    )
    generate.add_argument("--factor-iterations", type=int, default=8)
    generate.add_argument("--horizon", type=int, choices=(1, 2), default=2)

    transitions = commands.add_parser("generate-transitions")
    transitions.add_argument("--cases", required=True, type=Path)
    transitions.add_argument("--output", required=True, type=Path)
    transitions.add_argument("--dataset-id", required=True)
    transitions.add_argument("--video-skills-root", type=Path)
    transitions.add_argument("--no-video-skills-runtime", action="store_true")
    transitions.add_argument("--include-stop", action="store_true")
    transitions.add_argument("--allow-ai-provisional", action="store_true")
    transitions.add_argument(
        "--belief-backend", choices=("factor_graph", "factorized"), default="factor_graph"
    )
    transitions.add_argument("--factor-iterations", type=int, default=8)

    validate_transitions = commands.add_parser("validate-transitions")
    validate_transitions.add_argument("--dataset", required=True, type=Path)

    lock_transitions = commands.add_parser("lock-transitions")
    lock_transitions.add_argument("--dataset", required=True, type=Path)
    lock_transitions.add_argument("--output", required=True, type=Path)
    lock_transitions.add_argument("--annotator", required=True)
    lock_transitions.add_argument(
        "--status", choices=("ai_provisional", "human_locked"), required=True
    )

    export_transitions = commands.add_parser("export-transitions")
    export_transitions.add_argument("--dataset", required=True, type=Path)
    export_transitions.add_argument("--output", required=True, type=Path)
    export_transitions.add_argument("--allow-ai-provisional", action="store_true")

    packet = commands.add_parser("make-annotation")
    packet.add_argument("--runs-dir", required=True, type=Path)
    packet.add_argument("--output", required=True, type=Path)
    packet.add_argument("--packet-id", required=True)
    packet.add_argument("--max-comparisons-per-case", type=int, default=12)

    lock_packet = commands.add_parser("lock-annotation")
    lock_packet.add_argument("--packet", required=True, type=Path)
    lock_packet.add_argument("--output", required=True, type=Path)
    lock_packet.add_argument("--annotator", required=True)
    lock_packet.add_argument(
        "--status", choices=("ai_provisional", "human_locked"), required=True
    )

    export = commands.add_parser("export-training")
    export.add_argument("--packet", required=True, type=Path)
    export.add_argument("--output", required=True, type=Path)
    export.add_argument("--allow-ai-provisional", action="store_true")

    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--cases", required=True, type=Path)
    evaluate.add_argument("--output", required=True, type=Path)
    evaluate.add_argument("--allow-ai-provisional", action="store_true")
    evaluate.add_argument(
        "--gtsam-closed-loop",
        action="store_true",
        help="Require executed-read categorical correction through GTSAM; no fallback.",
    )

    train = commands.add_parser("train")
    train.add_argument("--training-jsonl", required=True, type=Path)
    train.add_argument("--output-dir", required=True, type=Path)
    train.add_argument("--allow-ai-provisional", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "mine-cases":
        overlay_paths = [path.expanduser().resolve() for path in args.overlay]
        if args.overlay_root is not None:
            overlay_paths.extend(
                sorted(args.overlay_root.expanduser().resolve().glob(args.glob))
            )
        overlay_paths = list(dict.fromkeys(overlay_paths))
        if not overlay_paths:
            raise ValueError("mine-cases requires --overlay-root or at least one --overlay")
        case_set, report = mine_navigation_cases(
            overlay_paths,
            case_set_id=args.case_set_id,
            desired_count=args.desired_count,
            per_video_limit=args.per_video_limit,
        )
        _write_json(args.output, case_set)
        _write_json(args.report, report)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0
    if args.command == "validate-cases":
        payload = _read_json(args.cases)
        errors = validate_navigation_case_set(payload)
        print(json.dumps({"valid": not errors, "errors": errors}, indent=2))
        return 0 if not errors else 1
    if args.command == "lock-cases":
        payload = lock_navigation_case_set(
            _read_json(args.cases),
            annotation_status=args.status,
            annotator=args.annotator,
        )
        _write_json(args.output, payload)
        return 0
    if args.command == "generate":
        manifest = generate_case_runs(
            args.cases,
            args.output_dir,
            belief_backend=args.belief_backend,
            factor_iterations=args.factor_iterations,
            horizon=args.horizon,
            video_skills_root=args.video_skills_root,
            use_video_skills_runtime=not args.no_video_skills_runtime,
        )
        print(json.dumps(manifest, indent=2))
        return 0
    if args.command == "generate-transitions":
        case_path = args.cases.expanduser().resolve()
        dataset = build_executed_transition_dataset(
            _read_json(case_path),
            case_root=case_path.parent,
            dataset_id=args.dataset_id,
            belief_backend=args.belief_backend,
            factor_iterations=args.factor_iterations,
            executor_factory=lambda: VideoSkillsL2Adapter(
                args.video_skills_root,
                use_video_skills_runtime=not args.no_video_skills_runtime,
            ),
            include_stop=args.include_stop,
            allow_ai_provisional=args.allow_ai_provisional,
            execution_mode=(
                "persisted_replay"
                if args.no_video_skills_runtime
                else "video_skills_runtime"
            ),
        )
        _write_json(args.output, dataset)
        print(json.dumps(dataset["summary"], indent=2, ensure_ascii=False))
        return 0
    if args.command == "validate-transitions":
        errors = validate_executed_transition_dataset(_read_json(args.dataset))
        print(json.dumps({"valid": not errors, "errors": errors}, indent=2))
        return 0 if not errors else 1
    if args.command == "lock-transitions":
        dataset = lock_executed_transition_dataset(
            _read_json(args.dataset),
            annotation_status=args.status,
            annotator=args.annotator,
        )
        _write_json(args.output, dataset)
        return 0
    if args.command == "export-transitions":
        records = export_executed_transition_training_records(
            _read_json(args.dataset),
            allow_ai_provisional=args.allow_ai_provisional,
        )
        _write_jsonl(args.output, records)
        print(json.dumps({"record_count": len(records)}, indent=2))
        return 0
    if args.command == "make-annotation":
        packet = make_annotation_packet(
            args.runs_dir,
            packet_id=args.packet_id,
            max_comparisons_per_case=args.max_comparisons_per_case,
        )
        _write_json(args.output, packet)
        print(json.dumps({"comparisons": len(packet["comparisons"])}, indent=2))
        return 0
    if args.command == "lock-annotation":
        packet = lock_annotation_packet(
            _read_json(args.packet),
            annotation_status=args.status,
            annotator=args.annotator,
        )
        _write_json(args.output, packet)
        return 0
    if args.command == "export-training":
        records = export_training_records(
            _read_json(args.packet),
            allow_ai_provisional=args.allow_ai_provisional,
        )
        _write_jsonl(args.output, records)
        print(json.dumps({"record_count": len(records)}, indent=2))
        return 0
    if args.command == "evaluate":
        case_path = args.cases.expanduser().resolve()
        report = evaluate_matched_navigation(
            _read_json(case_path),
            case_root=case_path.parent,
            allow_ai_provisional=args.allow_ai_provisional,
            gtsam_closed_loop=args.gtsam_closed_loop,
        )
        _write_json(args.output, report)
        print(json.dumps(report["strategies"], indent=2))
        return 0
    if args.command == "train":
        records = [
            json.loads(line)
            for line in args.training_jsonl.expanduser().resolve().read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        report = train_baselines(
            records,
            output_dir=args.output_dir,
            allow_ai_provisional=args.allow_ai_provisional,
        )
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


def generate_case_runs(
    case_path: Path,
    output_dir: Path,
    *,
    belief_backend: str = "factor_graph",
    factor_iterations: int = 8,
    horizon: int = 2,
    video_skills_root: Path | None = None,
    use_video_skills_runtime: bool = True,
) -> dict[str, Any]:
    source = case_path.expanduser().resolve()
    payload = _read_json(source)
    require_valid_navigation_case_set(payload)
    root = output_dir.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    used_names: set[str] = set()
    for case in payload["cases"]:
        case_id = str(case["case_id"])
        directory_name = _safe_name(case_id)
        if directory_name in used_names:
            raise ValueError(f"case directory collision after sanitizing: {case_id}")
        used_names.add(directory_name)
        overlay_path = _resolve_path(source.parent, str(case["overlay_path"]))
        loaded = load_overlay_artifact(overlay_path)
        if loaded.overlay.overlay_id != case["overlay_id"]:
            raise ValueError(f"case {case_id} overlay_id does not match its artifact")
        run_dir = root / directory_name
        arguments = [
            "--overlay", str(overlay_path),
            "--question", str(case["question"]),
            "--graph-read-budget", str(case["graph_read_budget"]),
            "--belief-backend", belief_backend,
            "--factor-iterations", str(factor_iterations),
            "--horizon", str(horizon),
            "--output-dir", str(run_dir),
        ]
        for seed in case["seed_event_ids"]:
            arguments.extend(("--seed-event", str(seed)))
        for role in case["missing_roles"]:
            arguments.extend(("--missing-role", str(role)))
        if video_skills_root is not None:
            arguments.extend(("--video-skills-root", str(video_skills_root)))
        if not use_video_skills_runtime:
            arguments.append("--no-video-skills-runtime")
        status = run_one(arguments)
        if status != 0:
            raise RuntimeError(f"navigation run failed for case {case_id}: {status}")
        rows.append(
            {
                "case_id": case_id,
                "directory": directory_name,
                "overlay_path": str(overlay_path),
                "sibling_artifact": str(run_dir / "sibling_checkpoint.json"),
                "summary": str(run_dir / "run_summary.json"),
            }
        )
    manifest = {
        "schema_version": "steam-preference-case-runs/v0.1",
        "case_set_id": payload["case_set_id"],
        "case_set_annotation_status": payload["annotation_status"],
        "belief_backend": belief_backend,
        "factor_iterations": factor_iterations,
        "horizon": horizon,
        "case_count": len(rows),
        "runs": rows,
    }
    _write_json(root / "batch_manifest.json", manifest)
    return manifest


def make_annotation_packet(
    runs_dir: Path,
    *,
    packet_id: str,
    max_comparisons_per_case: int | None = 12,
) -> dict[str, Any]:
    root = runs_dir.expanduser().resolve()
    manifest = _read_json(root / "batch_manifest.json")
    artifacts = []
    for row in manifest.get("runs") or []:
        path = Path(str(row["sibling_artifact"]))
        artifacts.append((str(row["case_id"]), _read_json(path)))
    return build_preference_annotation_packet(
        artifacts,
        packet_id=packet_id,
        max_comparisons_per_case=max_comparisons_per_case,
    )


def _safe_name(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9._-]+", "_", value).strip("._")
    if not normalized or normalized in {".", ".."}:
        raise ValueError(f"case_id cannot form a safe directory name: {value!r}")
    return normalized


def _resolve_path(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _write_json(path: Path, payload: Any) -> None:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
