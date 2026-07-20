"""CLI for preference-only navigation over one persisted L1.5 overlay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .artifacts import belief_to_dict, navigation_run_to_dict
from .belief import FactorizedBeliefBackend
from .factor_graph import FactorGraphBeliefBackend, GTSAM_AVAILABLE
from .overlay_io import load_overlay_artifact
from .planner import ClosedLoopNavigator, PreferenceOnlyPlanner
from .siblings import generate_sibling_artifact
from .video_skills_adapter import (
    VideoSkillsL2Adapter,
    build_video_skills_l2_rollout,
)
from .world_model import (
    RuleBasedObservationBeliefModel,
    RuleBasedTrajectoryPreferenceModel,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run preference-only closed-loop navigation on one overlay",
    )
    parser.add_argument("--overlay", required=True, type=Path)
    parser.add_argument("--question", required=True)
    parser.add_argument("--seed-event", action="append", default=[])
    parser.add_argument("--missing-role", action="append", default=None)
    parser.add_argument("--graph-read-budget", type=int, default=8)
    parser.add_argument(
        "--belief-backend",
        choices=("factor_graph", "factorized"),
        default="factor_graph",
    )
    parser.add_argument("--factor-iterations", type=int, default=8)
    parser.add_argument("--horizon", type=int, choices=(1, 2), default=2)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--max-second-actions", type=int, default=4)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--video-skills-root", type=Path)
    parser.add_argument(
        "--no-video-skills-runtime",
        action="store_true",
        help="Use persisted graph reads while retaining L2-compatible records",
    )
    parser.add_argument(
        "--skip-siblings",
        action="store_true",
        help="Do not execute all admissible actions from the initial checkpoint",
    )
    parser.add_argument("--require-embedding-files", action="store_true")
    parser.add_argument("--verify-embedding-checksums", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    loaded = load_overlay_artifact(
        args.overlay,
        require_embedding_files=args.require_embedding_files,
        verify_embedding_checksums=args.verify_embedding_checksums,
    )
    overlay = loaded.overlay
    backend = (
        FactorGraphBeliefBackend(inference_iterations=args.factor_iterations)
        if args.belief_backend == "factor_graph"
        else FactorizedBeliefBackend()
    )
    belief = backend.initialize(
        args.question,
        overlay,
        seed_evidence=tuple(args.seed_event),
        missing_roles=(
            tuple(args.missing_role) if args.missing_role is not None else None
        ),
        graph_read_budget=args.graph_read_budget,
    )
    world_model = RuleBasedObservationBeliefModel()
    preference_model = RuleBasedTrajectoryPreferenceModel()
    planner = PreferenceOnlyPlanner(
        world_model,
        preference_model,
        horizon=args.horizon,
        max_second_actions=args.max_second_actions,
    )
    executor = VideoSkillsL2Adapter(
        args.video_skills_root,
        use_video_skills_runtime=not args.no_video_skills_runtime,
    )

    sibling_artifact = None
    if not args.skip_siblings:
        sibling_artifact = generate_sibling_artifact(
            belief,
            overlay,
            backend=backend,
            executor=executor,
            world_model=world_model,
            preference_model=preference_model,
        )
    run = ClosedLoopNavigator(backend, planner, executor).run(
        belief,
        overlay,
        max_steps=args.max_steps,
    )
    l2_rollout = build_video_skills_l2_rollout(
        run,
        overlay,
        question=args.question,
    )

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "navigation_run.json", navigation_run_to_dict(run))
    _write_json(output_dir / "l2_rollout.json", l2_rollout)
    _write_jsonl(
        output_dir / "belief_snapshots.jsonl",
        [belief_to_dict(snapshot) for snapshot in run.belief_snapshots],
    )
    if sibling_artifact is not None:
        _write_json(output_dir / "sibling_checkpoint.json", sibling_artifact)
    summary = {
        "schema_version": "steam-preference-navigation-run-summary/v0.1",
        "source_overlay": str(loaded.source_path),
        "overlay_id": overlay.overlay_id,
        "example_id": overlay.example_id,
        "embedding_problems": list(loaded.embedding_problems),
        "embedding_model": _embedding_model(overlay),
        "belief_backend": run.final_belief.backend_name,
        "belief_backend_ref": run.final_belief.backend_ref,
        "gtsam_available": GTSAM_AVAILABLE,
        "step_count": len(run.steps),
        "real_observation_count": sum(len(step.observation_ids) for step in run.steps),
        "final_answerability": run.final_belief.answerability.value,
        "final_missing_roles": list(run.final_belief.missing_roles),
        "preference_output_contract": "ordinal_only",
        "sibling_branch_count": (
            len(sibling_artifact["branches"])
            if sibling_artifact is not None
            else 0
        ),
        "sibling_annotation_status": (
            sibling_artifact["annotation_status"]
            if sibling_artifact is not None
            else "not_generated"
        ),
    }
    _write_json(output_dir / "run_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def _embedding_model(overlay: Any) -> str | None:
    return next(
        (
            node.embedding_ref.model
            for node in overlay.atomic_events + overlay.l1_observations
            if node.embedding_ref is not None
        ),
        None,
    )


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
