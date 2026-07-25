"""Resumable case-isolated launcher for the frozen multi-trajectory cohort."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

from .closed_loop import action_divergence
from .multi_trajectory_cgbench import MULTI_ARMS, _aggregate


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _slug(case_id: str) -> str:
    digest = hashlib.sha256(case_id.encode("utf-8")).hexdigest()[:12]
    return f"case_{digest}"


def _complete(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        value = _read(path)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError):
        return False
    selected_case_ids = value.get("selected_case_ids") or ()
    runs = value.get("runs") or ()
    if (
        value.get("errors")
        or not isinstance(selected_case_ids, list)
        or len(selected_case_ids) != 1
        or not isinstance(runs, list)
        or len(runs) != len(MULTI_ARMS)
        or set(value.get("arms") or ()) != set(MULTI_ARMS)
    ):
        return False
    case_id = str(selected_case_ids[0])
    run_keys = [
        (str(run.get("case_id") or ""), str(run.get("arm") or ""))
        for run in runs
        if isinstance(run, dict)
    ]
    return len(run_keys) == len(runs) and set(run_keys) == {
        (case_id, arm) for arm in MULTI_ARMS
    }


def _select_case_ids(
    gate: dict[str, Any],
    requested_case_ids: list[str],
    case_limit: int | None,
) -> list[str]:
    runnable = list(gate.get("runnable_case_ids") or ())
    if requested_case_ids:
        if len(requested_case_ids) != len(set(requested_case_ids)):
            raise ValueError("--case-id values must be unique")
        unknown = set(requested_case_ids) - set(runnable)
        if unknown:
            raise ValueError(
                "requested cases are not runnable under the frozen gate: "
                + ", ".join(sorted(unknown))
            )
        requested = set(requested_case_ids)
        selected = [case_id for case_id in runnable if case_id in requested]
    else:
        selected = runnable
    if case_limit is not None:
        selected = selected[:case_limit]
    if not selected:
        raise ValueError("no runnable cases selected")
    return selected


def _validate_caption_candidate_mode(
    gate: dict[str, Any],
    *,
    disabled: bool,
) -> None:
    modes = {
        bool(row.get("caption_candidate_overlay_loaded"))
        for row in gate.get("graphs") or ()
        if row.get("graph_available")
    }
    if len(modes) > 1:
        raise ValueError("frozen gate mixes caption-candidate graph modes")
    if modes and (not disabled) != next(iter(modes)):
        expected = "enabled" if next(iter(modes)) else "disabled"
        raise ValueError(
            "caption-candidate runtime mode does not match frozen gate; "
            f"expected {expected}"
        )


def _run_case(args: argparse.Namespace, case_id: str) -> dict[str, Any]:
    slug = _slug(case_id)
    case_dir = args.output_dir / "cases"
    output = case_dir / f"{slug}.json"
    log = case_dir / f"{slug}.log"
    if not args.force and _complete(output):
        return {"case_id": case_id, "status": "reused", "output": str(output)}
    command = [
        sys.executable,
        "-m",
        (
            "steam_video_new.implicit_world_model.full_graph_iwm."
            "multi_trajectory_cgbench"
        ),
        "--dataset",
        str(args.dataset),
        "--hidden-key",
        str(args.hidden_key),
        "--selection",
        str(args.selection),
        "--graph-root",
        str(args.graph_root),
        "--compiled-gate",
        str(args.compiled_gate),
        "--output",
        str(output),
        "--keys-py",
        str(args.keys_py),
        "--mode",
        "run",
        "--model",
        args.model,
        "--capacity",
        str(args.capacity),
        "--graph-read-budget",
        str(args.read_budget),
        "--case-limit",
        "1",
        "--case-id",
        case_id,
        "--transition-batch-size",
        str(args.transition_batch_size),
        "--comparison-batch-size",
        str(args.comparison_batch_size),
        "--max-complete-pairs",
        str(args.max_complete_pairs),
        "--question-role-cache",
        str(case_dir / f"{slug}.roles.json"),
        "--response-cache",
        str(case_dir / f"{slug}.responses.json"),
        "--cache-mode",
        "record",
        "--timeout-s",
        str(args.timeout_s),
        "--max-tokens",
        str(args.max_tokens),
        "--reasoning-effort",
        args.reasoning_effort,
    ]
    if args.disable_caption_candidates:
        command.append("--disable-caption-candidates")
    completed = subprocess.run(
        command,
        cwd=args.repo_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(completed.stdout, encoding="utf-8")
    return {
        "case_id": case_id,
        "status": "completed" if completed.returncode == 0 else "failed",
        "returncode": completed.returncode,
        "output": str(output),
        "log": str(log),
    }


def _combine(
    output_dir: Path,
    case_ids: list[str],
    statuses: list[dict[str, Any]],
) -> dict[str, Any]:
    runs: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    method_failures: list[dict[str, Any]] = []
    completed_case_ids: list[str] = []
    for case_id in case_ids:
        output = output_dir / "cases" / f"{_slug(case_id)}.json"
        if not output.is_file():
            continue
        value = _read(output)
        errors.extend(value.get("errors") or ())
        method_failures.extend(value.get("method_failures") or ())
        if _complete(output):
            completed_case_ids.append(case_id)
            runs.extend(value.get("runs") or ())
    by_case: dict[str, dict[str, dict[str, Any]]] = {}
    for run in runs:
        by_case.setdefault(str(run["case_id"]), {})[str(run["arm"])] = run
    divergences: list[dict[str, Any]] = []
    for case_id, case_runs in by_case.items():
        reference = case_runs.get("world_model_guided")
        if reference is None:
            continue
        for arm, candidate in case_runs.items():
            if arm != "world_model_guided":
                divergences.append(
                    {"case_id": case_id, **action_divergence(reference, candidate)}
                )
    return {
        "schema_version": "steam-multi-trajectory-cohort-launch/v0.1",
        "requested_case_ids": case_ids,
        "completed_case_ids": completed_case_ids,
        "pending_case_ids": [
            case_id for case_id in case_ids if case_id not in completed_case_ids
        ],
        "statuses": statuses,
        "runs": runs,
        "errors": errors,
        "method_failures": method_failures,
        "metrics_by_arm": _aggregate(runs, MULTI_ARMS),
        "action_divergence": divergences,
        "training_performed": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--hidden-key", required=True, type=Path)
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--graph-root", required=True, type=Path)
    parser.add_argument("--compiled-gate", required=True, type=Path)
    parser.add_argument("--keys-py", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model", default="qwen/qwen3.6-flash")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--capacity", type=int, default=256)
    parser.add_argument("--read-budget", type=int, default=2)
    parser.add_argument(
        "--transition-batch-size",
        type=int,
        default=1,
        help="Number of compact shared-action groups per model request.",
    )
    parser.add_argument("--comparison-batch-size", type=int, default=12)
    parser.add_argument("--max-complete-pairs", type=int, default=4096)
    parser.add_argument("--timeout-s", type=int, default=180)
    parser.add_argument("--max-tokens", type=int, default=8000)
    parser.add_argument(
        "--reasoning-effort", choices=("low", "medium", "high"), default="low"
    )
    parser.add_argument(
        "--disable-caption-candidates",
        action="store_true",
        help=(
            "Do not load optional caption-candidate overlays. Required when the "
            "frozen compile gate was built without them."
        ),
    )
    parser.add_argument(
        "--case-id",
        action="append",
        default=[],
        help=(
            "Run one explicitly frozen case ID; repeat for multiple cases. "
            "Order remains the frozen gate order."
        ),
    )
    parser.add_argument("--case-limit", type=int)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    gate = _read(args.compiled_gate)
    if not gate.get("gate_passed"):
        raise ValueError("compiled gate did not pass")
    _validate_caption_candidate_mode(
        gate,
        disabled=args.disable_caption_candidates,
    )
    case_ids = _select_case_ids(gate, args.case_id, args.case_limit)
    statuses: list[dict[str, Any]] = []
    executor = ThreadPoolExecutor(max_workers=args.workers)
    futures: dict[Any, str] = {}
    try:
        futures = {
            executor.submit(_run_case, args, case_id): case_id for case_id in case_ids
        }
        for future in as_completed(futures):
            status = future.result()
            statuses.append(status)
            combined = _combine(args.output_dir, case_ids, statuses)
            _write(args.output_dir / "cohort.progress.json", combined)
            print(
                json.dumps(
                    {
                        "case_id": status["case_id"],
                        "status": status["status"],
                        "completed": len(combined["completed_case_ids"]),
                        "pending": len(combined["pending_case_ids"]),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    except KeyboardInterrupt:
        for future in futures:
            future.cancel()
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    else:
        executor.shutdown(wait=True)
    combined = _combine(args.output_dir, case_ids, statuses)
    _write(args.output_dir / "cohort.final.json", combined)
    return 0 if not combined["pending_case_ids"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
