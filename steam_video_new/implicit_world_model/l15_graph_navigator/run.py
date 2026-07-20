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
from .context import ReasoningContextBuilder
from .contracts import ReasoningContextBudget
from .interventions import FrozenBeliefWorldModel, TransitionIntervention
from .gpt_oss import (
    DEFAULT_GPT_OSS_MODEL,
    GPTOSSObservationBeliefModel,
    GPTOSSTrajectoryPreferenceModel,
    OpenAICompatibleCategoricalClient,
    OPENROUTER_API_BASE,
)
from .siblings import generate_sibling_artifact
from .video_skills_adapter import (
    VideoSkillsL2Adapter,
    build_video_skills_l2_rollout,
)
from .world_model import (
    RuleBasedObservationBeliefModel,
    RuleBasedTrajectoryPreferenceModel,
)
from factor_graph.correction_policy import CorrectionMode


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
    parser.add_argument(
        "--belief-mode",
        choices=tuple(mode.value for mode in CorrectionMode),
        default=CorrectionMode.IWM_BELIEF_ONLY.value,
        help=(
            "Main IWM belief, categorical-triggered GTSAM backup, or "
            "GTSAM-always baseline. No mode silently falls back."
        ),
    )
    parser.add_argument(
        "--belief-checkpoint-in",
        type=Path,
        help="Resume an audited GTSAM backup checkpoint.",
    )
    parser.add_argument(
        "--belief-checkpoint-out",
        type=Path,
        help="Write the final audited GTSAM backup checkpoint.",
    )
    parser.add_argument("--factor-iterations", type=int, default=8)
    parser.add_argument("--horizon", type=int, choices=(1, 2), default=2)
    parser.add_argument(
        "--transition-intervention",
        choices=tuple(value.value for value in TransitionIntervention),
        default=TransitionIntervention.NORMAL.value,
    )
    parser.add_argument(
        "--freeze-world-model",
        action="store_true",
        help="Predict every hop from the initial belief checkpoint",
    )
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--max-second-actions", type=int, default=4)
    parser.add_argument("--context-max-nodes", type=int, default=24)
    parser.add_argument("--context-max-edges", type=int, default=32)
    parser.add_argument("--candidate-hop-budget", type=int, default=8)
    parser.add_argument("--comparison-budget", type=int, default=32)
    parser.add_argument("--recent-hop-window", type=int, default=3)
    parser.add_argument(
        "--reasoning-model-backend",
        choices=("rule", "gpt-oss-120b"),
        default="rule",
    )
    parser.add_argument("--reasoning-model", default=DEFAULT_GPT_OSS_MODEL)
    parser.add_argument("--reasoning-api-base")
    parser.add_argument(
        "--reasoning-keys-py",
        type=Path,
        help="Python file defining OPENROUTER_API_KEY",
    )
    parser.add_argument("--reasoning-api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--reasoning-timeout-s", type=int, default=180)
    parser.add_argument("--reasoning-max-tokens", type=int, default=1600)
    parser.add_argument(
        "--reasoning-effort",
        choices=("low", "medium", "high"),
        default="low",
    )
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
    belief_mode = CorrectionMode(args.belief_mode)
    if belief_mode is CorrectionMode.IWM_BELIEF_ONLY:
        backend = (
            FactorGraphBeliefBackend(inference_iterations=args.factor_iterations)
            if args.belief_backend == "factor_graph"
            else FactorizedBeliefBackend()
        )
    else:
        from factor_graph.navigation_backend import GTSAMExecutedReadBeliefBackend

        backend = GTSAMExecutedReadBeliefBackend(
            mode=(
                "backup"
                if belief_mode is CorrectionMode.IWM_WITH_GTSAM_BACKUP
                else "correct"
            )
        )
    if args.belief_checkpoint_in is not None:
        restore = getattr(backend, "restore_checkpoint", None)
        if not callable(restore):
            raise ValueError("belief checkpoint input requires a GTSAM belief mode")
        belief = restore(
            json.loads(args.belief_checkpoint_in.read_text(encoding="utf-8")),
            overlay,
        )
        if belief.question != args.question:
            raise ValueError("checkpoint question does not match --question")
    else:
        belief = backend.initialize(
            args.question,
            overlay,
            seed_evidence=tuple(args.seed_event),
            missing_roles=(
                tuple(args.missing_role) if args.missing_role is not None else None
            ),
            graph_read_budget=args.graph_read_budget,
        )
    reasoning_client: OpenAICompatibleCategoricalClient | None = None
    if args.reasoning_model_backend == "gpt-oss-120b":
        reasoning_client = (
            OpenAICompatibleCategoricalClient.from_openrouter_keys_file(
                args.reasoning_keys_py,
                api_base=(
                    args.reasoning_api_base
                    or OPENROUTER_API_BASE
                ),
                model=args.reasoning_model,
                timeout_s=args.reasoning_timeout_s,
                max_tokens=args.reasoning_max_tokens,
                reasoning_effort=args.reasoning_effort,
            )
            if args.reasoning_keys_py is not None
            else OpenAICompatibleCategoricalClient.from_environment(
                api_base=args.reasoning_api_base,
                model=args.reasoning_model,
                api_key_env=args.reasoning_api_key_env,
                timeout_s=args.reasoning_timeout_s,
                max_tokens=args.reasoning_max_tokens,
                reasoning_effort=args.reasoning_effort,
            )
        )
        world_model = GPTOSSObservationBeliefModel(reasoning_client)
        preference_model = GPTOSSTrajectoryPreferenceModel(reasoning_client)
    else:
        world_model = RuleBasedObservationBeliefModel()
        preference_model = RuleBasedTrajectoryPreferenceModel()
    if args.freeze_world_model:
        world_model = FrozenBeliefWorldModel(world_model)
    context_budget = ReasoningContextBudget(
        max_nodes=args.context_max_nodes,
        max_edges=args.context_max_edges,
        max_candidate_hops=args.candidate_hop_budget,
        max_comparisons=args.comparison_budget,
        recent_hop_window=args.recent_hop_window,
    )
    planner = PreferenceOnlyPlanner(
        world_model,
        preference_model,
        horizon=args.horizon,
        max_second_actions=args.max_second_actions,
        context_builder=ReasoningContextBuilder(context_budget),
        transition_intervention=TransitionIntervention(
            args.transition_intervention
        ),
    )
    context_builder = planner.context_builder
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
            context_builder=context_builder,
            label_source=(
                "gpt-oss-120b_categorical_provisional/v0.1"
                if args.reasoning_model_backend == "gpt-oss-120b"
                else "rule_based_provisional/v0.1"
            ),
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
    if args.belief_checkpoint_out is not None:
        export_checkpoint = getattr(backend, "export_checkpoint", None)
        if not callable(export_checkpoint):
            raise ValueError("belief checkpoint output requires a GTSAM belief mode")
        _write_json(args.belief_checkpoint_out, export_checkpoint(run.final_belief))
    correction_audits = [
        step.belief_update_audit
        for step in run.steps
        if step.belief_update_audit is not None
    ]
    summary = {
        "schema_version": "steam-preference-navigation-run-summary/v0.2",
        "source_overlay": str(loaded.source_path),
        "overlay_id": overlay.overlay_id,
        "example_id": overlay.example_id,
        "embedding_problems": list(loaded.embedding_problems),
        "embedding_model": _embedding_model(overlay),
        "belief_backend": run.final_belief.backend_name,
        "belief_mode": belief_mode.value,
        "belief_backend_ref": run.final_belief.backend_ref,
        "gtsam_available": GTSAM_AVAILABLE,
        "gtsam_backup_activation_count": sum(
            (audit.get("backup_trigger") or {}).get("activate") is True
            for audit in correction_audits
        ),
        "gtsam_factor_activation_count": sum(
            audit.get("factor_activated") is True for audit in correction_audits
        ),
        "step_count": len(run.steps),
        "real_observation_count": sum(len(step.observation_ids) for step in run.steps),
        "final_answerability": run.final_belief.answerability.value,
        "final_missing_roles": list(run.final_belief.missing_roles),
        "preference_output_contract": "ordinal_only",
        "reasoning_model_backend": args.reasoning_model_backend,
        "transition_intervention": args.transition_intervention,
        "world_model_frozen": args.freeze_world_model,
        "reasoning_model": (
            args.reasoning_model
            if args.reasoning_model_backend == "gpt-oss-120b"
            else "rule_based_categorical_baseline"
        ),
        "reasoning_provider": (
            "openrouter"
            if args.reasoning_keys_py is not None
            else (
                "openai_compatible"
                if args.reasoning_model_backend == "gpt-oss-120b"
                else "offline_rule"
            )
        ),
        "reasoning_transport_audit": (
            {
                "request_count": len(reasoning_client.response_audits),
                "finish_reasons": [
                    row.get("finish_reason")
                    for row in reasoning_client.response_audits
                ],
                "prompt_tokens": [
                    row.get("prompt_tokens")
                    for row in reasoning_client.response_audits
                ],
                "completion_tokens": [
                    row.get("completion_tokens")
                    for row in reasoning_client.response_audits
                ],
            }
            if reasoning_client is not None
            else None
        ),
        "reasoning_context_budget": {
            "max_nodes": context_budget.max_nodes,
            "max_edges": context_budget.max_edges,
            "max_candidate_hops": context_budget.max_candidate_hops,
            "max_comparisons": context_budget.max_comparisons,
            "recent_hop_window": context_budget.recent_hop_window,
        },
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
