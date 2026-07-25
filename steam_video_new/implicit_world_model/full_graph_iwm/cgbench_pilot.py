"""Fixed-case CG-Bench gate and matched-budget full-graph IWM pilot."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable

from steam_video_new.implicit_world_model.l15_graph_navigator.gpt_oss import (
    DEFAULT_GPT_OSS_MODEL,
    OpenAICompatibleCategoricalClient,
)
from steam_video_new.implicit_world_model.l15_graph_navigator.overlay_io import (
    load_overlay_artifact,
)

from .action_compiler import GraphActionCompiler, visible_graph_nodes
from .caption_candidates import augment_graph_with_caption_candidates
from .closed_loop import (
    ClueInterval,
    action_divergence,
    graph_fingerprint,
    run_oracle_clue_ceiling,
    run_real_read_closed_loop,
)
from .contracts import (
    ActionKind,
    CursorBeliefState,
    FullGraphPlanDecision,
    RetainedEvidenceGraph,
)
from .gpt_oss import (
    GPTOSSFullGraphPreferenceModel,
    GPTOSSQuestionBeliefInitializer,
    GPTOSSRealEvidenceBeliefUpdater,
    GPTOSSFullGraphSetwisePreferenceModel,
    GPTOSSFullGraphWorldModel,
)
from .graph_adapter import build_l1_l15_navigation_graph
from .interventions import FrozenWorldModel, ShuffledWorldModel
from .localization import GPTOSSEntryLocalizer
from .model_input import build_iwm_graph_input, graph_input_to_categorical_payload
from .planner import FullGraphIWMPlanner
from .reactive import GPTOSSReactiveGraphPlanner
from .transition_cache import (
    PersistentCategoricalResponseCacheClient,
    PersistentQuestionRoleCache,
    PersistentTransitionCacheWorldModel,
)


GATE_SCHEMA = "steam-full-graph-iwm-cgbench-gate/v0.1"
PILOT_SCHEMA = "steam-full-graph-iwm-cgbench-pilot/v0.1"
FORBIDDEN_PLANNER_KEYS = {
    "answer",
    "answer_key",
    "answer_text",
    "clue_intervals",
    "clue_indices",
    "hop_alignment",
    "temporal_gt_relation",
}
ARM_ORDER = (
    "world_model_guided",
    "no_world_model",
    "shuffled_world_model_prediction",
    "frozen_world_model",
    "immediate_effect_only",
    "oracle_clue_ceiling",
)


class _FailClosedAbstainPlanner:
    """Represent entry-localization contract failure without dropping a case."""

    def plan(
        self,
        belief: CursorBeliefState,
        graph: RetainedEvidenceGraph,
    ) -> FullGraphPlanDecision:
        actions = GraphActionCompiler().compile(belief, graph)
        selected = next(
            action for action in actions if action.kind is ActionKind.ABSTAIN
        )
        return FullGraphPlanDecision(
            selected_action=selected,
            planning_status="entry_localization_failed_fail_closed",
            trajectories=(),
            preferences=(),
            undominated_trajectory_ids=(),
            legal_action_count=len(actions),
            top_k_applied=False,
        )


def compile_cgbench_gate(
    *,
    dataset: dict[str, Any],
    hidden_key: dict[str, Any],
    selection: dict[str, Any],
    graph_root: Path,
    capacity: int = 64,
    read_budget: int = 8,
    video_limit: int = 8,
    cases_per_video: int = 1,
    include_caption_candidates: bool = True,
    allowed_case_ids: Iterable[str] = (),
) -> dict[str, Any]:
    """Compile frozen graphs before any IWM service call.

    Video and case ordering comes only from the public selection/dataset.  GT is
    joined after compilation to compute evaluator-only clue retention.
    """

    if capacity < 1 or read_budget < 1 or video_limit < 1 or cases_per_video < 1:
        raise ValueError("capacity, read budget, and selection limits must be positive")
    public_cases = list(dataset.get("cases") or [])
    if _contains_key(dataset, FORBIDDEN_PLANNER_KEYS):
        # The grounded dataset legitimately contains offline transition targets;
        # only the whitelisted planner fields below may cross the boundary.
        dataset_boundary = "whitelist_required_and_applied"
    else:
        dataset_boundary = "public_artifact_contains_no_hidden_fields"
    hidden_by_case = {str(row["case_id"]): row for row in hidden_key.get("cases") or []}
    allowed = set(str(case_id) for case_id in allowed_case_ids)
    cases_by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in public_cases:
        cases_by_video[str(case["video_id"])].append(case)
    allowed_video_ids = {
        str(row["video_id"])
        for row in public_cases
        if not allowed or str(row["case_id"]) in allowed
    }
    requested = [
        row
        for row in selection.get("videos") or []
        if str(row["video_id"]) in allowed_video_ids
    ][:video_limit]
    graph_root = graph_root.expanduser().resolve()
    graph_rows: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []
    graph_cache: dict[str, RetainedEvidenceGraph] = {}
    for entry in requested:
        video_id = str(entry["video_id"])
        graph_path = graph_root / video_id / "causal_temporal_overlay.json"
        graph_row: dict[str, Any] = {
            "video_id": video_id,
            "source_graph": str(graph_path),
            "graph_available": graph_path.is_file(),
            "compile_status": "graph_missing",
        }
        if graph_path.is_file():
            try:
                loaded = load_overlay_artifact(graph_path, validate_schema=True)
                graph = _build_graph_with_optional_caption_candidates(
                    loaded.overlay,
                    sample_dir=graph_path.parent,
                    capacity=capacity,
                    include_caption_candidates=include_caption_candidates,
                )
                question_independent = bool(
                    loaded.overlay.metadata.get("question_independent_contract")
                    or loaded.overlay.metadata.get("question_independent")
                )
                graph_cache[video_id] = graph
                correlation_build = graph.metadata.get("correlation_build") or {}
                graph_row.update(
                    {
                        "compile_status": "compiled",
                        "question_independent": question_independent,
                        "retained_node_count": len(graph.nodes),
                        "temporal_edge_count": len(graph.temporal_edges),
                        "correlation_edge_count": len(graph.correlation_edges),
                        "candidate_edge_count": len(graph.candidate_edges),
                        "caption_candidate_overlay_loaded": (
                            "caption_candidate_overlay" in graph.metadata
                        ),
                        "verified_relation_count": len(graph.verified_relations),
                        "soft_correlation_build_available": (
                            correlation_build.get("status")
                            != "unavailable_missing_embeddings"
                        ),
                        "capacity": graph.capacity,
                        "graph_fingerprint": graph_fingerprint(graph),
                        "observation_end_s": graph.metadata.get("observation_end_s"),
                    }
                )
            except Exception as exc:
                graph_row.update(
                    {
                        "compile_status": "compile_error",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
        graph_rows.append(graph_row)

        video_cases = sorted(
            cases_by_video.get(video_id, []), key=lambda row: str(row["case_id"])
        )
        selected_cases = (
            [row for row in video_cases if str(row["case_id"]) in allowed]
            if allowed
            else video_cases[:cases_per_video]
        )
        for case in selected_cases:
            case_id = str(case["case_id"])
            row: dict[str, Any] = {
                "case_id": case_id,
                "video_id": video_id,
                "split": case.get("split"),
                "graph_compile_status": graph_row["compile_status"],
                "runnable": False,
            }
            hidden = hidden_by_case.get(case_id)
            if hidden is None:
                row["status"] = "hidden_key_missing"
                case_rows.append(row)
                continue
            clues = _clues(hidden)
            graph = graph_cache.get(video_id)
            horizon = graph_row.get("observation_end_s")
            within_horizon = graph is not None and (
                horizon is None
                or all(clue.end_s <= float(horizon) + 1e-6 for clue in clues)
            )
            row.update(
                {
                    "clue_count_evaluator_only": len(clues),
                    "all_clues_within_observation_horizon": within_horizon,
                    "all_clues_within_graph_horizon": within_horizon,
                }
            )
            if graph is None:
                row["status"] = "graph_unavailable"
                case_rows.append(row)
                continue
            audit_entry_ids = tuple(node.node_id for node in graph.nodes[:1])
            belief = CursorBeliefState(
                belief_id=f"belief:{case_id}:gate",
                question=str(case["planner_input"]["question"]),
                localized_entry_node_ids=audit_entry_ids,
                remaining_reads=read_budget,
            )
            actions = GraphActionCompiler().compile(belief, graph)
            model_input = build_iwm_graph_input(belief, graph, actions)
            payload = graph_input_to_categorical_payload(model_input)
            leaked_keys = sorted(_find_keys(payload, FORBIDDEN_PLANNER_KEYS))
            unread_leaks = sum(
                view.evidence_value is not None
                for view in model_input.nodes
                if not view.acquired
            )
            visible_node_ids = {view.key.node_id for view in model_input.nodes}
            start_targets = {
                action.target_id
                for action in actions
                if action.kind.value == "start_at"
            }
            covered = _covered_clue_indices(graph, clues)
            oracle_covered = _maximum_clue_coverage_at_two_hop_budget(
                graph,
                clues,
                min(read_budget, 2),
            )
            row.update(
                {
                    "status": "compiled",
                    "legal_action_count": len(actions),
                    "visible_node_count": len(visible_node_ids),
                    "structural_audit_entry_count": len(audit_entry_ids),
                    "global_start_at_all_nodes_disabled": True,
                    "all_visible_nodes_have_start_action": (
                        start_targets == visible_node_ids
                    ),
                    "top_k_applied": False,
                    "unread_evidence_value_leak_count": unread_leaks,
                    "forbidden_planner_keys": leaked_keys,
                    "retained_clue_count_evaluator_only": len(covered),
                    "retained_clue_recall_evaluator_only": (len(covered) / len(clues)),
                    "oracle_clue_count_at_planning_horizon_evaluator_only": oracle_covered,
                    "oracle_clue_recall_at_planning_horizon_evaluator_only": (
                        oracle_covered / len(clues)
                    ),
                    "oracle_clue_complete_at_planning_horizon_evaluator_only": (
                        oracle_covered == len(clues)
                    ),
                    "graph_fingerprint": graph_fingerprint(graph),
                }
            )
            row["runnable"] = bool(
                within_horizon
                and len(covered) == len(clues)
                and not leaked_keys
                and unread_leaks == 0
                and start_targets == visible_node_ids
                and graph_row.get("question_independent") is True
            )
            row["scientifically_runnable"] = bool(
                row["runnable"]
                and row["oracle_clue_complete_at_planning_horizon_evaluator_only"]
            )
            case_rows.append(row)

    checks = {
        "all_requested_graphs_available": bool(requested)
        and all(row["graph_available"] for row in graph_rows),
        "all_available_graphs_compile": bool(requested)
        and all(row["compile_status"] == "compiled" for row in graph_rows),
        "all_graphs_question_independent": bool(requested)
        and all(row.get("question_independent") is True for row in graph_rows),
        "all_graphs_have_embedding_correlation_build": bool(requested)
        and all(
            row.get("soft_correlation_build_available") is True for row in graph_rows
        ),
        "planner_input_has_no_hidden_keys": all(
            not row.get("forbidden_planner_keys")
            for row in case_rows
            if row.get("status") == "compiled"
        ),
        "planner_input_has_no_unread_values": all(
            row.get("unread_evidence_value_leak_count") == 0
            for row in case_rows
            if row.get("status") == "compiled"
        ),
        "eligible_case_exists": any(
            row.get("all_clues_within_graph_horizon") is True for row in case_rows
        ),
        "every_horizon_eligible_case_retains_all_clues": all(
            row.get("runnable") is True
            for row in case_rows
            if row.get("all_clues_within_graph_horizon") is True
        ),
        "scientifically_runnable_case_exists": any(
            row.get("scientifically_runnable") is True for row in case_rows
        ),
    }
    gate_passed = all(checks.values())
    return {
        "schema_version": GATE_SCHEMA,
        "dataset_id": dataset.get("dataset_id"),
        "selection_schema": selection.get("schema_version"),
        "graph_root": str(graph_root),
        "capacity": capacity,
        "read_budget": read_budget,
        "requested_video_count": len(requested),
        "selected_case_count": len(case_rows),
        "fixed_cohort_case_filter_applied": bool(allowed),
        "allowed_case_count": len(allowed) if allowed else None,
        "runnable_case_ids": [
            row["case_id"] for row in case_rows if row.get("runnable") is True
        ],
        "scientifically_runnable_case_ids": [
            row["case_id"]
            for row in case_rows
            if row.get("scientifically_runnable") is True
        ],
        "dataset_boundary": dataset_boundary,
        "checks": checks,
        "gate_passed": gate_passed,
        "gate_contract": (
            "structural, clue-retention, and exact oracle-at-read-budget preflight; "
            "navigation metrics remain separate and are never collapsed into this boolean"
        ),
        "graphs": graph_rows,
        "cases": case_rows,
        "gpt_service_called": False,
        "training_performed": False,
    }


def run_gpt_oss_matched_pilot(
    *,
    gate: dict[str, Any],
    dataset: dict[str, Any],
    hidden_key: dict[str, Any],
    graph_root: Path,
    client: OpenAICompatibleCategoricalClient,
    capacity: int,
    read_budget: int,
    case_limit: int = 8,
    max_trajectory_pairs: int = 4096,
    max_imagined_transition_requests: int = 512,
    rollout_horizon: int = 1,
    arms: Iterable[str] = ARM_ORDER,
    setwise_preference: bool = False,
    case_ids: Iterable[str] = (),
    execute_stable_ties: bool = False,
    transition_cache_path: Path | None = None,
    transition_cache_mode: str = "record",
    include_caption_candidates: bool = True,
    world_model_batch_size: int = 48,
    world_model_max_contexts_per_batch: int = 8,
    question_role_cache_path: Path | None = None,
    response_cache_path: Path | None = None,
    progress_path: Path | None = None,
    resume_progress: bool = False,
) -> dict[str, Any]:
    """Run matched graph/read-budget arms after a successful gate."""

    if not gate.get("gate_passed"):
        raise ValueError("CG-Bench compile gate did not pass; GPT calls are blocked")
    if rollout_horizon not in {1, 2}:
        raise ValueError("rollout_horizon must be one or two")
    requested_arms = tuple(dict.fromkeys(arms))
    unknown = set(requested_arms) - set(ARM_ORDER)
    if unknown:
        raise ValueError(f"unsupported pilot arms: {sorted(unknown)}")
    public_by_case = {str(row["case_id"]): row for row in dataset.get("cases") or []}
    hidden_by_case = {str(row["case_id"]): row for row in hidden_key.get("cases") or []}
    requested_case_ids = set(case_ids)
    selected_case_ids = [
        case_id
        for case_id in gate.get("runnable_case_ids") or []
        if not requested_case_ids or case_id in requested_case_ids
    ][:case_limit]
    missing_requested = requested_case_ids - set(selected_case_ids)
    if missing_requested:
        raise ValueError(
            f"requested cases are not runnable under the gate: {sorted(missing_requested)}"
        )
    runs: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    method_failures: list[dict[str, str]] = []
    if resume_progress:
        if progress_path is None or not progress_path.is_file():
            raise FileNotFoundError(
                "--resume-progress requires an existing progress file"
            )
        prior = _read_json(progress_path)
        if prior.get("selected_case_ids") != selected_case_ids:
            raise ValueError("progress case selection does not match this run")
        if prior.get("requested_arms") != list(requested_arms):
            raise ValueError("progress arm selection does not match this run")
        runs = list(prior.get("runs") or [])
        errors = list(prior.get("errors") or [])
        method_failures = list(prior.get("method_failures") or [])
    response_audit_start = len(client.response_audits)

    def checkpoint(*, complete: bool = False) -> None:
        if progress_path is None:
            return
        _write_json(
            progress_path,
            {
                "schema_version": f"{PILOT_SCHEMA}/progress",
                "complete": complete,
                "selected_case_ids": selected_case_ids,
                "requested_arms": list(requested_arms),
                "completed_run_count": len(runs),
                "error_count": len(errors),
                "method_failure_count": len(method_failures),
                "runs": runs,
                "errors": errors,
                "method_failures": method_failures,
                "metrics_by_arm": _aggregate_metrics(runs, requested_arms),
                "model_response_count": (
                    len(client.response_audits) - response_audit_start
                ),
                "training_performed": False,
            },
        )

    checkpoint()
    belief_initializer: Any = GPTOSSQuestionBeliefInitializer(client)
    entry_localizer = GPTOSSEntryLocalizer(client)
    if question_role_cache_path is not None:
        belief_initializer = PersistentQuestionRoleCache(
            belief_initializer,
            question_role_cache_path,
            mode=transition_cache_mode,
        )
    for case_id in selected_case_ids:
        public = public_by_case[case_id]
        hidden = hidden_by_case[case_id]
        video_id = str(public["video_id"])
        graph_path = (
            graph_root.expanduser().resolve()
            / video_id
            / "causal_temporal_overlay.json"
        )
        loaded = load_overlay_artifact(graph_path, validate_schema=True)
        graph = _build_graph_with_optional_caption_candidates(
            loaded.overlay,
            sample_dir=graph_path.parent,
            capacity=capacity,
            include_caption_candidates=include_caption_candidates,
        )
        expected_fingerprint = next(
            row["graph_fingerprint"]
            for row in gate["cases"]
            if row["case_id"] == case_id
        )
        if graph_fingerprint(graph) != expected_fingerprint:
            raise ValueError(f"graph changed after gate for {case_id}")
        clues = _clues(hidden)
        question = str(public["planner_input"]["question"])
        initial_missing_roles: tuple[str, ...] = ()
        initial_entry_node_ids: tuple[str, ...] = ()
        entry_localization_evaluator_only: dict[str, Any] | None = None
        entry_localization_failure: str | None = None
        if any(arm != "oracle_clue_ceiling" for arm in requested_arms):
            try:
                initial_missing_roles = belief_initializer.initialize(question)
                initial_entry_node_ids = entry_localizer.localize(
                    question=question,
                    missing_roles=initial_missing_roles,
                    graph=graph,
                )
                localized_clues = _covered_clue_indices_by_node_ids(
                    graph, clues, set(initial_entry_node_ids)
                )
                entry_localization_evaluator_only = {
                    "covered_clue_count": len(localized_clues),
                    "clue_count": len(clues),
                    "clue_recall": (
                        len(localized_clues) / len(clues) if clues else None
                    ),
                    "fed_back_to_localizer_or_planner": False,
                }
            except Exception as exc:
                entry_localization_failure = f"{type(exc).__name__}: {exc}"
                method_failures.append(
                    {
                        "case_id": case_id,
                        "stage": "entry_localization",
                        "failure": entry_localization_failure,
                    }
                )
                entry_localizer.audits.append(
                    {
                        "candidate_address_count": len(visible_graph_nodes(graph)),
                        "selected_anchor_count": 0,
                        "selected_node_ids": [],
                        "status": "failed_contract",
                        "top_k_applied": False,
                        "numeric_score_used": False,
                        "failure": entry_localization_failure,
                        "fail_closed": True,
                    }
                )
        for arm in requested_arms:
            if any(
                row.get("case_id") == case_id and row.get("arm") == arm for row in runs
            ):
                continue
            try:
                if arm == "oracle_clue_ceiling":
                    run = run_oracle_clue_ceiling(
                        case_id=case_id,
                        question=question,
                        graph=graph,
                        clue_intervals=clues,
                        read_budget=read_budget,
                    )
                elif entry_localization_failure is not None:
                    run = run_real_read_closed_loop(
                        case_id=case_id,
                        question=question,
                        graph=graph,
                        clue_intervals=clues,
                        planner=_FailClosedAbstainPlanner(),
                        read_budget=read_budget,
                        arm=arm,
                        initial_missing_roles=initial_missing_roles,
                        initial_entry_node_ids=(),
                    )
                    run["method_audit"] = {
                        "planner_type": "FailClosedEntryLocalizationBoundary",
                        "entry_localization_failed": True,
                        "failure": entry_localization_failure,
                        "top_k_applied": False,
                    }
                    run["entry_localization"] = entry_localizer.audits[-1]
                    run["entry_localization_evaluator_only"] = None
                else:
                    planner = _planner_for_arm(
                        arm,
                        client,
                        rollout_horizon=rollout_horizon,
                        max_trajectory_pairs=max_trajectory_pairs,
                        max_imagined_transition_requests=(
                            max_imagined_transition_requests
                        ),
                        setwise_preference=setwise_preference,
                        execute_stable_ties=execute_stable_ties,
                        transition_cache_path=transition_cache_path,
                        transition_cache_mode=transition_cache_mode,
                        world_model_batch_size=world_model_batch_size,
                        world_model_max_contexts_per_batch=(
                            world_model_max_contexts_per_batch
                        ),
                    )
                    run = run_real_read_closed_loop(
                        case_id=case_id,
                        question=question,
                        graph=graph,
                        clue_intervals=clues,
                        planner=planner,
                        read_budget=read_budget,
                        arm=arm,
                        initial_missing_roles=initial_missing_roles,
                        initial_entry_node_ids=initial_entry_node_ids,
                        belief_updater=GPTOSSRealEvidenceBeliefUpdater(client),
                    )
                    run["method_audit"] = _planner_method_audit(planner)
                    run["entry_localization"] = entry_localizer.audits[-1]
                    run["entry_localization_evaluator_only"] = (
                        entry_localization_evaluator_only
                    )
                if run["graph_fingerprint"] != expected_fingerprint:
                    raise ValueError("matched arm mutated the retained graph")
                run["split"] = public.get("split")
                runs.append(run)
                checkpoint()
            except Exception as exc:
                failure = f"{type(exc).__name__}: {exc}"
                errors.append(
                    {
                        "case_id": case_id,
                        "arm": arm,
                        "error": failure,
                    }
                )
                if arm != "oracle_clue_ceiling":
                    run = run_real_read_closed_loop(
                        case_id=case_id,
                        question=question,
                        graph=graph,
                        clue_intervals=clues,
                        planner=_FailClosedAbstainPlanner(),
                        read_budget=read_budget,
                        arm=arm,
                        initial_missing_roles=initial_missing_roles,
                        initial_entry_node_ids=(),
                    )
                    run["method_audit"] = {
                        "planner_type": "FailClosedRuntimeBoundary",
                        "arm_runtime_failed": True,
                        "failure": failure,
                        "top_k_applied": False,
                    }
                    run["entry_localization"] = entry_localizer.audits[-1]
                    run["entry_localization_evaluator_only"] = (
                        entry_localization_evaluator_only
                    )
                    run["split"] = public.get("split")
                    runs.append(run)
                checkpoint()
    divergences = []
    by_case: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for run in runs:
        by_case[run["case_id"]][run["arm"]] = run
    for case_id, case_runs in by_case.items():
        reference = case_runs.get("world_model_guided")
        if reference is None:
            continue
        for arm, run in case_runs.items():
            if arm != "world_model_guided":
                divergences.append(
                    {"case_id": case_id, **action_divergence(reference, run)}
                )
    result = {
        "schema_version": PILOT_SCHEMA,
        "dataset_id": dataset.get("dataset_id"),
        "gate_schema": gate.get("schema_version"),
        "model": client.model,
        "model_role": "inference_and_data_gathering_only",
        "rollout_horizon": rollout_horizon,
        "arms": list(requested_arms),
        "matched_contract": {
            "same_graph_fingerprint": True,
            "same_legal_action_compiler": True,
            "same_read_budget": read_budget,
            "same_capacity": capacity,
            "top_k_applied": False,
            "hidden_clue_feedback_to_planner": False,
            "imagined_belief_used_as_real_belief": False,
            "preference_mode": "setwise_categorical"
            if setwise_preference
            else "pairwise_categorical",
            "stable_tie_execution_requested": execute_stable_ties,
            "stable_tie_execution_effective": False,
            "transition_cache": (
                {
                    "path": str(transition_cache_path.expanduser().resolve()),
                    "mode": transition_cache_mode,
                    "hidden_clue_or_answer_used": False,
                }
                if transition_cache_path is not None
                else None
            ),
            "caption_candidate_overlay_enabled": include_caption_candidates,
            "world_model_transport_batch_size": world_model_batch_size,
            "world_model_max_contexts_per_batch": (world_model_max_contexts_per_batch),
            "max_imagined_transition_requests": (max_imagined_transition_requests),
            "question_role_cache": (
                str(question_role_cache_path.expanduser().resolve())
                if question_role_cache_path is not None
                else None
            ),
            "categorical_response_cache": (
                {
                    "path": str(response_cache_path.expanduser().resolve()),
                    "mode": transition_cache_mode,
                    "request_payload_stored": False,
                    "hidden_clue_or_answer_used": False,
                }
                if response_cache_path is not None
                else None
            ),
        },
        "runs": runs,
        "errors": errors,
        "method_failures": method_failures,
        "metrics_by_arm": _aggregate_metrics(runs, requested_arms),
        "metrics_by_split_and_arm": {
            split: _aggregate_metrics(
                [run for run in runs if run.get("split") == split],
                requested_arms,
            )
            for split in sorted(
                {str(run["split"]) for run in runs if run.get("split") is not None}
            )
        },
        "paired_arm_effects": _paired_arm_effects(runs),
        "transition_outcome_confusion_by_arm": (
            _transition_outcome_confusion(runs, requested_arms)
        ),
        "action_divergence": divergences,
        "model_response_audits": client.response_audits[response_audit_start:],
        "entry_localization_audits": entry_localizer.audits,
        "metric_contract": (
            "coverage, read efficiency, action divergence, abstention, delayed "
            "success, and latency are separate; no lexicographic aggregate gate"
        ),
        "answer_accuracy_status": "not_evaluated_without_terminal_answer_head",
        "training_performed": False,
    }
    checkpoint(complete=True)
    return result


def _planner_method_audit(
    planner: FullGraphIWMPlanner | GPTOSSReactiveGraphPlanner,
) -> dict[str, Any]:
    """Expose method behavior without leaking hidden evaluator information."""

    audit: dict[str, Any] = {
        "planner_type": type(planner).__name__,
        "top_k_applied": False,
    }
    world_model = getattr(planner, "world_model", None)
    wrappers: list[dict[str, Any]] = []
    visited: set[int] = set()
    while world_model is not None and id(world_model) not in visited:
        visited.add(id(world_model))
        row: dict[str, Any] = {"type": type(world_model).__name__}
        cache_audits = getattr(world_model, "cache_audits", None)
        if cache_audits is not None:
            rows = list(cache_audits)
            row["cache_batches"] = len(rows)
            row["cache_hits"] = sum(int(value.get("hit_count", 0)) for value in rows)
            row["cache_misses"] = sum(int(value.get("miss_count", 0)) for value in rows)
        batch_audits = getattr(world_model, "batch_audits", None)
        if batch_audits is not None:
            rows = list(batch_audits)
            row["intervention_batches"] = len(rows)
            row["changed_descriptor_count"] = sum(
                int(value.get("changed_descriptor_count", 0)) for value in rows
            )
        normalization = getattr(world_model, "normalization_audits", None)
        if normalization is not None:
            row["categorical_normalization_count"] = len(normalization)
        wrappers.append(row)
        world_model = getattr(world_model, "delegate", None)
    if wrappers:
        audit["world_model_chain"] = wrappers
    preference = getattr(planner, "setwise_preference_model", None)
    if preference is not None:
        tournament_rows = list(getattr(preference, "tournament_audits", ()))
        audit["setwise_preference"] = {
            "protocol": "complete_coverage_categorical_tournament",
            "decision_count": len(tournament_rows),
            "decisions": tournament_rows,
        }
    return audit


def _planner_for_arm(
    arm: str,
    client: OpenAICompatibleCategoricalClient,
    *,
    rollout_horizon: int,
    max_trajectory_pairs: int,
    max_imagined_transition_requests: int = 512,
    setwise_preference: bool = False,
    execute_stable_ties: bool = False,
    transition_cache_path: Path | None = None,
    transition_cache_mode: str = "record",
    world_model_batch_size: int = 48,
    world_model_max_contexts_per_batch: int = 8,
) -> FullGraphIWMPlanner | GPTOSSReactiveGraphPlanner:
    if rollout_horizon not in {1, 2}:
        raise ValueError("rollout_horizon must be one or two")
    preference = GPTOSSFullGraphPreferenceModel(client)
    if arm == "no_world_model":
        return GPTOSSReactiveGraphPlanner(
            client,
            max_action_pairs=max_trajectory_pairs,
            setwise=setwise_preference,
            execute_stable_ties=execute_stable_ties,
        )
    base_world_model = GPTOSSFullGraphWorldModel(
        client,
        batch_size=world_model_batch_size,
        max_contexts_per_batch=world_model_max_contexts_per_batch,
    )
    if transition_cache_path is not None:
        base_world_model = PersistentTransitionCacheWorldModel(
            base_world_model,
            transition_cache_path,
            mode=transition_cache_mode,
        )
    if arm == "shuffled_world_model_prediction":
        world_model = ShuffledWorldModel(base_world_model)
        horizon = rollout_horizon
    elif arm == "frozen_world_model":
        world_model = FrozenWorldModel(base_world_model)
        horizon = rollout_horizon
    elif arm == "immediate_effect_only":
        world_model = base_world_model
        horizon = 1
    elif arm == "world_model_guided":
        world_model = base_world_model
        horizon = rollout_horizon
    else:
        raise ValueError(f"arm does not use the IWM planner: {arm}")
    return FullGraphIWMPlanner(
        world_model,
        preference,
        horizon=horizon,
        max_trajectory_pairs=max_trajectory_pairs,
        max_imagined_transition_requests=max_imagined_transition_requests,
        setwise_preference_model=(
            GPTOSSFullGraphSetwisePreferenceModel(client)
            if setwise_preference
            else None
        ),
        execute_stable_ties=execute_stable_ties,
    )


def _aggregate_metrics(
    runs: list[dict[str, Any]], arms: Iterable[str]
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for arm in arms:
        selected = [run for run in runs if run["arm"] == arm]
        metrics = [run["metrics"] for run in selected]
        frontier_rows = [
            row
            for run in selected
            if (row := _first_hop_frontier_audit(run)) is not None
        ]
        planning_statuses = [
            str(step.get("decision", {}).get("planning_status") or "")
            for run in selected
            for step in run.get("steps") or []
        ]
        legal_action_counts = [
            [
                int(step.get("decision", {}).get("legal_action_count"))
                for step in run.get("steps") or []
                if step.get("decision", {}).get("legal_action_count") is not None
            ]
            for run in selected
        ]
        localization_recalls = [
            (
                0.0
                if (run.get("method_audit") or {}).get("entry_localization_failed")
                else (run.get("entry_localization_evaluator_only") or {}).get(
                    "clue_recall"
                )
            )
            for run in selected
        ]
        localized_clue_available = [
            run
            for run, recall in zip(selected, localization_recalls)
            if recall is not None and float(recall) > 0.0
        ]
        result[arm] = {
            "case_count": len(selected),
            "clue_coverage_complete_rate": _mean_bool(
                row["clue_coverage_complete"] for row in metrics
            ),
            "mean_clue_recall": _mean(row["clue_recall"] for row in metrics),
            "mean_real_read_count": _mean(row["real_read_count"] for row in metrics),
            "mean_read_efficiency": _mean(row["read_efficiency"] for row in metrics),
            "abstain_rate": _mean_bool(row["abstained"] for row in metrics),
            "delayed_reasoning_success_rate": _mean_bool(
                row["delayed_reasoning_success"] for row in metrics
            ),
            "mean_latency_s": _mean(row["latency_s"] for row in metrics),
            "mean_entry_localization_clue_recall": _mean(localization_recalls),
            "entry_localization_any_clue_rate": (
                _mean_bool(
                    float(recall) > 0.0
                    for recall in localization_recalls
                    if recall is not None
                )
            ),
            "first_read_clue_hit_rate_when_localized_clue_available": (
                _mean_bool(
                    _first_real_read_hits_clue(run) for run in localized_clue_available
                )
            ),
            "mean_entry_anchor_count": _mean(
                (run.get("entry_localization") or {}).get("selected_anchor_count")
                for run in selected
            ),
            "entry_localization_failure_rate": _mean_bool(
                bool((run.get("method_audit") or {}).get("entry_localization_failed"))
                for run in selected
            ),
            "arm_runtime_failure_rate": _mean_bool(
                bool((run.get("method_audit") or {}).get("arm_runtime_failed"))
                for run in selected
            ),
            "mean_initial_legal_action_count": _mean(
                counts[0] if counts else None for counts in legal_action_counts
            ),
            "mean_max_local_legal_action_count": _mean(
                max(counts) if counts else None for counts in legal_action_counts
            ),
            "mean_initial_first_hop_count": _mean(
                row["initial_count"] for row in frontier_rows
            ),
            "mean_final_first_hop_frontier_count": _mean(
                row["final_count"] for row in frontier_rows
            ),
            "mean_first_hop_frontier_retention_rate": _mean(
                row["retention_rate"] for row in frontier_rows
            ),
            "rollout_budget_abstain_rate": (
                sum(
                    status == "abstain_first_hop_frontier_rollout_budget_exceeded"
                    for status in planning_statuses
                )
                / len(planning_statuses)
                if planning_statuses
                else None
            ),
            "answer_accuracy": None,
        }
    return result


def _first_real_read_hits_clue(run: dict[str, Any]) -> bool:
    for step in run.get("steps") or []:
        label = step.get("realized_label_evaluator_only")
        if isinstance(label, dict):
            return bool(label.get("newly_covered_clue_indices"))
    return False


def _paired_arm_effects(runs: list[dict[str, Any]]) -> dict[str, Any]:
    by_case: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for run in runs:
        by_case[str(run["case_id"])][str(run["arm"])] = run
    reference_arm = "world_model_guided"
    result: dict[str, Any] = {}
    for candidate_arm in (
        "no_world_model",
        "shuffled_world_model_prediction",
        "immediate_effect_only",
        "oracle_clue_ceiling",
    ):
        pairs = [
            (case_runs[reference_arm], case_runs[candidate_arm])
            for case_runs in by_case.values()
            if reference_arm in case_runs and candidate_arm in case_runs
        ]
        deltas = [
            float(reference["metrics"]["clue_recall"])
            - float(candidate["metrics"]["clue_recall"])
            for reference, candidate in pairs
        ]
        result[candidate_arm] = {
            "paired_case_count": len(pairs),
            "mean_clue_recall_delta_reference_minus_candidate": _mean(deltas),
            "reference_better_clue_recall_count": sum(delta > 0.0 for delta in deltas),
            "equal_clue_recall_count": sum(delta == 0.0 for delta in deltas),
            "reference_worse_clue_recall_count": sum(delta < 0.0 for delta in deltas),
            "reference_only_delayed_success_count": sum(
                bool(reference["metrics"]["delayed_reasoning_success"])
                and not bool(candidate["metrics"]["delayed_reasoning_success"])
                for reference, candidate in pairs
            ),
            "candidate_only_delayed_success_count": sum(
                bool(candidate["metrics"]["delayed_reasoning_success"])
                and not bool(reference["metrics"]["delayed_reasoning_success"])
                for reference, candidate in pairs
            ),
            "action_divergence_rate": _mean_bool(
                action_divergence(reference, candidate)["action_diverged"]
                for reference, candidate in pairs
            ),
        }
    return result


def _transition_outcome_confusion(
    runs: list[dict[str, Any]], arms: Iterable[str]
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for arm in arms:
        matrix: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        outcome_matches: list[bool] = []
        delta_matches: list[bool] = []
        for run in runs:
            if run.get("arm") != arm:
                continue
            for step in run.get("steps") or []:
                realized = step.get("realized_label_evaluator_only")
                if not isinstance(realized, dict):
                    continue
                predicted = _selected_initial_transition(step)
                if predicted is None:
                    continue
                predicted_outcome = str(
                    (predicted.get("observation") or {}).get("outcome") or "missing"
                )
                realized_outcome = str(realized.get("observation_outcome") or "missing")
                matrix[realized_outcome][predicted_outcome] += 1
                outcome_matches.append(predicted_outcome == realized_outcome)
                predicted_delta = {
                    key: value
                    for key, value in (predicted.get("belief_delta") or {}).items()
                    if key != "predicted_only"
                }
                realized_delta = {
                    key: value
                    for key, value in (realized.get("belief_delta") or {}).items()
                    if key != "predicted_only"
                }
                delta_matches.append(predicted_delta == realized_delta)
        result[arm] = {
            "evaluated_transition_count": sum(
                sum(row.values()) for row in matrix.values()
            ),
            "outcome_confusion_realized_to_predicted": {
                realized: dict(sorted(predicted.items()))
                for realized, predicted in sorted(matrix.items())
            },
            "exact_outcome_match_rate": _mean_bool(outcome_matches),
            "exact_belief_delta_match_rate": _mean_bool(delta_matches),
        }
    return result


def _selected_initial_transition(
    step: dict[str, Any],
) -> dict[str, Any] | None:
    selected_id = str((step.get("selected_action") or {}).get("action_id") or "")
    if not selected_id:
        return None
    decision = step.get("decision") or {}
    for trajectory in decision.get("initial_trajectories") or []:
        transitions = trajectory.get("transitions") or []
        if not transitions:
            continue
        transition = transitions[0]
        if str((transition.get("action") or {}).get("action_id") or "") == selected_id:
            return transition
    return None


def _first_hop_frontier_audit(run: dict[str, Any]) -> dict[str, float] | None:
    preference = (run.get("method_audit") or {}).get("setwise_preference") or {}
    decisions = preference.get("decisions") or []
    if not decisions:
        return None
    rounds = decisions[0].get("rounds") or []
    if not rounds:
        return None
    initial = int(rounds[0]["input_count"])
    final = int(rounds[-1]["survivor_count"])
    return {
        "initial_count": float(initial),
        "final_count": float(final),
        "retention_rate": final / initial if initial else 0.0,
    }


def _covered_clue_indices(
    graph: RetainedEvidenceGraph, clues: tuple[ClueInterval, ...]
) -> set[int]:
    return {
        index
        for index, clue in enumerate(clues)
        if any(
            node.time_span.start_s < clue.end_s and clue.start_s < node.time_span.end_s
            for node in graph.nodes
        )
    }


def _covered_clue_indices_by_node_ids(
    graph: RetainedEvidenceGraph,
    clues: tuple[ClueInterval, ...],
    node_ids: set[str],
) -> set[int]:
    return {
        index
        for index, clue in enumerate(clues)
        if any(
            node.node_id in node_ids
            and node.time_span.start_s < clue.end_s
            and clue.start_s < node.time_span.end_s
            for node in graph.nodes
        )
    }


def _maximum_clue_coverage_at_two_hop_budget(
    graph: RetainedEvidenceGraph,
    clues: tuple[ClueInterval, ...],
    planning_horizon: int,
) -> int:
    """Compute the exact ceiling under the implemented one/two-hop topology."""

    if planning_horizon not in {1, 2}:
        raise ValueError("planning horizon must be one or two")
    masks: dict[str, int] = {}
    for node in graph.nodes:
        mask = 0
        for index, clue in enumerate(clues):
            if (
                node.time_span.start_s < clue.end_s
                and clue.start_s < node.time_span.end_s
            ):
                mask |= 1 << index
        if mask:
            masks[node.node_id] = mask
    best = max((mask.bit_count() for mask in masks.values()), default=0)
    if planning_horizon == 1:
        return best

    adjacency: dict[str, set[str]] = {node.node_id: set() for node in graph.nodes}
    for edge in graph.temporal_edges:
        adjacency[edge.src].add(edge.dst)
        adjacency[edge.dst].add(edge.src)
    for edge in graph.correlation_edges:
        if edge.evidence_refs and edge.src_to_dst_affinity > 0.0:
            adjacency[edge.src].add(edge.dst)
        if edge.evidence_refs and edge.dst_to_src_affinity > 0.0:
            adjacency[edge.dst].add(edge.src)
    for edge in graph.candidate_edges:
        if edge.permits(edge.src, edge.dst):
            adjacency[edge.src].add(edge.dst)
        if edge.permits(edge.dst, edge.src):
            adjacency[edge.dst].add(edge.src)
    for source_id, source_mask in masks.items():
        for target_id in adjacency[source_id]:
            best = max(
                best,
                (source_mask | masks.get(target_id, 0)).bit_count(),
            )
    return best


def _build_graph_with_optional_caption_candidates(
    overlay: Any,
    *,
    sample_dir: Path,
    capacity: int,
    include_caption_candidates: bool = True,
) -> RetainedEvidenceGraph:
    graph = build_l1_l15_navigation_graph(overlay, capacity=capacity)
    if not include_caption_candidates:
        return graph
    candidate_path = sample_dir / "l1_l15_caption_candidates.json"
    if not candidate_path.is_file():
        return graph
    artifact = _read_json(candidate_path)
    return augment_graph_with_caption_candidates(graph, artifact)


def _clues(hidden_case: dict[str, Any]) -> tuple[ClueInterval, ...]:
    return tuple(
        ClueInterval(float(row["start_s"]), float(row["end_s"]))
        for row in hidden_case.get("clue_intervals") or []
    )


def _mean(values: Iterable[float | int | None]) -> float | None:
    selected = [float(value) for value in values if value is not None]
    return fmean(selected) if selected else None


def _mean_bool(values: Iterable[bool]) -> float | None:
    selected = list(values)
    return fmean(float(value) for value in selected) if selected else None


def _contains_key(value: Any, forbidden: set[str]) -> bool:
    return bool(_find_keys(value, forbidden))


def _find_keys(value: Any, forbidden: set[str]) -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key).casefold() in forbidden:
                found.add(str(key))
            found.update(_find_keys(child, forbidden))
    elif isinstance(value, list):
        for child in value:
            found.update(_find_keys(child, forbidden))
    return found


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--hidden-key", required=True, type=Path)
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--graph-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--progress-output",
        type=Path,
        help="Atomic per-arm progress journal for long model-backed runs.",
    )
    parser.add_argument("--resume-progress", action="store_true")
    parser.add_argument("--capacity", type=int, default=64)
    parser.add_argument("--graph-read-budget", type=int, default=8)
    parser.add_argument("--video-limit", type=int, default=8)
    parser.add_argument("--cases-per-video", type=int, default=1)
    parser.add_argument("--case-limit", type=int, default=8)
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--max-trajectory-pairs", type=int, default=4096)
    parser.add_argument("--max-imagined-transition-requests", type=int, default=512)
    parser.add_argument("--rollout-horizon", type=int, choices=(1, 2), default=1)
    parser.add_argument(
        "--mode", choices=("compile-gate", "gpt-oss-120b"), default="compile-gate"
    )
    parser.add_argument("--keys-py", type=Path)
    parser.add_argument("--model", default=DEFAULT_GPT_OSS_MODEL)
    parser.add_argument("--timeout-s", type=int, default=180)
    parser.add_argument("--max-tokens", type=int, default=8000)
    parser.add_argument(
        "--reasoning-effort", choices=("low", "medium", "high"), default="low"
    )
    parser.add_argument("--setwise-preference", action="store_true")
    parser.add_argument("--execute-stable-ties", action="store_true")
    parser.add_argument("--disable-caption-candidates", action="store_true")
    parser.add_argument("--transition-cache", type=Path)
    parser.add_argument("--response-cache", type=Path)
    parser.add_argument("--question-role-cache", type=Path)
    parser.add_argument(
        "--fixed-cohort-gate",
        type=Path,
        help=(
            "Passed fixed-cohort gate whose locked case IDs define the only "
            "cases eligible for the matched run."
        ),
    )
    parser.add_argument(
        "--transition-cache-mode", choices=("record", "replay"), default="record"
    )
    parser.add_argument("--world-model-batch-size", type=int, default=48)
    parser.add_argument("--world-model-max-contexts-per-batch", type=int, default=8)
    parser.add_argument(
        "--arm",
        action="append",
        choices=ARM_ORDER,
        help="Repeat to run a subset; default runs every matched arm.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    dataset = _read_json(args.dataset)
    hidden = _read_json(args.hidden_key)
    selection = _read_json(args.selection)
    allowed_case_ids: tuple[str, ...] = ()
    if args.fixed_cohort_gate is not None:
        fixed_gate = _read_json(args.fixed_cohort_gate)
        if fixed_gate.get("gate_passed") is not True:
            raise ValueError("--fixed-cohort-gate must be a passed gate")
        fixed_capacity = fixed_gate.get("compile_capacity")
        if fixed_capacity is not None and int(fixed_capacity) != args.capacity:
            raise ValueError("fixed-cohort gate capacity does not match --capacity")
        locked_case_ids = {str(value) for value in fixed_gate["locked_case_ids"]}
        if args.case_id:
            unknown = set(args.case_id) - locked_case_ids
            if unknown:
                raise ValueError(
                    "requested cases are not locked by fixed cohort gate: "
                    f"{sorted(unknown)}"
                )
            allowed_case_ids = tuple(args.case_id)
        else:
            allowed_case_ids = tuple(sorted(locked_case_ids))
    gate = compile_cgbench_gate(
        dataset=dataset,
        hidden_key=hidden,
        selection=selection,
        graph_root=args.graph_root,
        capacity=args.capacity,
        read_budget=args.graph_read_budget,
        video_limit=args.video_limit,
        cases_per_video=args.cases_per_video,
        include_caption_candidates=not args.disable_caption_candidates,
        allowed_case_ids=allowed_case_ids,
    )
    if args.mode == "compile-gate":
        _write_json(args.output, gate)
        print(json.dumps(gate["checks"], indent=2))
        return 0 if gate["gate_passed"] else 2
    if args.keys_py is None:
        raise ValueError("--mode gpt-oss-120b requires --keys-py")
    if not gate["gate_passed"]:
        _write_json(args.output, gate)
        print(json.dumps(gate["checks"], indent=2))
        return 2
    client = OpenAICompatibleCategoricalClient.from_openrouter_keys_file(
        args.keys_py,
        model=args.model,
        timeout_s=args.timeout_s,
        max_tokens=args.max_tokens,
        reasoning_effort=args.reasoning_effort,
    )
    if args.response_cache is not None:
        client = PersistentCategoricalResponseCacheClient(
            client,
            args.response_cache,
            mode=args.transition_cache_mode,
        )
    question_role_cache = args.question_role_cache
    if question_role_cache is None and args.transition_cache is not None:
        question_role_cache = args.transition_cache.with_suffix(".roles.json")
    pilot = run_gpt_oss_matched_pilot(
        gate=gate,
        dataset=dataset,
        hidden_key=hidden,
        graph_root=args.graph_root,
        client=client,
        capacity=args.capacity,
        read_budget=args.graph_read_budget,
        case_limit=args.case_limit,
        max_trajectory_pairs=args.max_trajectory_pairs,
        max_imagined_transition_requests=args.max_imagined_transition_requests,
        rollout_horizon=args.rollout_horizon,
        arms=args.arm or ARM_ORDER,
        setwise_preference=args.setwise_preference,
        case_ids=args.case_id,
        execute_stable_ties=args.execute_stable_ties,
        transition_cache_path=args.transition_cache,
        transition_cache_mode=args.transition_cache_mode,
        include_caption_candidates=not args.disable_caption_candidates,
        world_model_batch_size=args.world_model_batch_size,
        world_model_max_contexts_per_batch=(args.world_model_max_contexts_per_batch),
        question_role_cache_path=question_role_cache,
        response_cache_path=args.response_cache,
        progress_path=(
            args.progress_output
            if args.progress_output is not None
            else args.output.with_suffix(args.output.suffix + ".progress.json")
        ),
        resume_progress=args.resume_progress,
    )
    pilot["compile_gate"] = gate
    _write_json(args.output, pilot)
    print(json.dumps(pilot["metrics_by_arm"], indent=2))
    return 0 if not pilot["errors"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
