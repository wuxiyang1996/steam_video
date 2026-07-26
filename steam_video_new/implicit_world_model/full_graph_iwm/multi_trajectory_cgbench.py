"""Matched-budget CG-Bench runner for the multi-trajectory IWM main method."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
from pathlib import Path
import json
from statistics import fmean
from typing import Any, Iterable, Sequence

from steam_video_new.implicit_world_model.l15_graph_navigator.gpt_oss import (
    DEFAULT_GPT_OSS_MODEL,
    OpenAICompatibleCategoricalClient,
)
from steam_video_new.implicit_world_model.l15_graph_navigator.overlay_io import (
    load_overlay_artifact,
)

from .action_compiler import GraphActionCompiler
from .cgbench_pilot import (
    _build_graph_with_optional_caption_candidates,
    _clues,
    compile_cgbench_gate,
)
from .closed_loop import (
    HiddenClueCoverageEvaluator,
    action_divergence,
    graph_fingerprint,
    run_oracle_clue_ceiling,
)
from .contracts import ActionKind, CursorBeliefState, RetainedEvidenceGraph
from .gpt_oss import GPTOSSQuestionBeliefInitializer, GPTOSSRealEvidenceBeliefUpdater
from .gtsam_backup import GTSAMMultiTrajectoryBeliefUpdater
from .localization import GPTOSSEntryLocalizer
from .multi_trajectory import (
    GPTOSSRealTrajectoryEvidenceAssessor,
    MultiTrajectoryPlanDecision,
    TrajectoryExpansion,
    TrajectoryPool,
    TrajectoryStatus,
    initialize_trajectory_pool,
    run_multi_trajectory_closed_loop,
    shared_action_key,
)
from .multi_trajectory_rollout import (
    GPTOSSCategoricalMultiTrajectoryModel,
    MultiTrajectoryRolloutPlanner,
    ReactiveMultiTrajectoryPlanner,
    ShuffledHypothesisWorldModel,
)
from .real_evidence import ArtifactBackedRealEvidenceReader
from .transition_cache import (
    PersistentCategoricalResponseCacheClient,
    PersistentQuestionRoleCache,
)


MULTI_PILOT_SCHEMA = "steam-multi-trajectory-iwm-cgbench-pilot/v0.2"
MULTI_RUNTIME_CONTRACT = "steam-multi-trajectory-runtime/v1.0"
MULTI_ARMS = (
    "world_model_guided",
    "no_world_model",
    "shuffled_world_model_prediction",
    "immediate_effect_only",
    "oracle_clue_ceiling",
)


@dataclass(frozen=True)
class TerminalAnswerDecision:
    status: str
    selected_choice: str | None
    rationale: str


@dataclass(frozen=True)
class EvidenceSufficiencyDecision:
    status: str
    rationale: str
    evaluator_only: bool = True
    fed_back_to_planner: bool = False


class GPTOSSEvidenceSufficiencyEvaluator:
    """Judge grounded evidence after execution; never participate in planning."""

    def __init__(self, client: Any) -> None:
        self.client = client

    def evaluate(
        self,
        *,
        question: str,
        choices: Sequence[str],
        evaluator_answer: str,
        observations: Sequence[Any],
    ) -> EvidenceSufficiencyDecision:
        aliases = {_alias("choice", index): row for index, row in enumerate(choices)}
        answer_alias = next(
            (alias for alias, value in aliases.items() if value == evaluator_answer),
            None,
        )
        if answer_alias is None:
            raise ValueError("evaluator answer is absent from public choices")
        result = self.client.complete_json(
            task=(
                "As an evaluator only, decide whether the acquired grounded evidence "
                "is sufficient to distinguish the reference answer from alternatives."
            ),
            payload={
                "question": question,
                "choices": aliases,
                "reference_answer": answer_alias,
                "acquired_grounded_evidence": [
                    _real_observation_payload(row) for row in observations
                ],
                "allowed_status": [
                    "supports_reference",
                    "supports_alternative",
                    "insufficient",
                    "inconclusive",
                ],
                "required_output": {"only_keys": ["status", "rationale"]},
                "contract": {
                    "evaluation_only": True,
                    "not_available_to_planner": True,
                    "judge_direct_evidence_not_temporal_overlap": True,
                    "categorical_only": True,
                },
            },
        )
        if not isinstance(result, dict) or set(result) != {"status", "rationale"}:
            raise ValueError("evidence-sufficiency response schema is invalid")
        if _contains_number(result):
            raise ValueError("evidence-sufficiency response contains a numeric value")
        status = str(result["status"])
        if status not in {
            "supports_reference",
            "supports_alternative",
            "insufficient",
            "inconclusive",
        }:
            raise ValueError("evidence-sufficiency status is invalid")
        return EvidenceSufficiencyDecision(status, str(result["rationale"] or ""))


class GPTOSSMultiTrajectoryAnswerSelector:
    """Select an answer categorically from real evidence and final trajectories."""

    def __init__(self, client: Any) -> None:
        self.client = client
        self.model_name = str(getattr(client, "model", "openai/gpt-oss-120b"))

    def select(
        self,
        pool: TrajectoryPool,
        choices: Sequence[str],
        graph: RetainedEvidenceGraph,
        *,
        real_observations: Sequence[Any] = (),
    ) -> TerminalAnswerDecision:
        if not choices or len(choices) != len(set(choices)):
            raise ValueError("answer choices must be non-empty and unique")
        choice_aliases = {
            _alias("choice", index): value for index, value in enumerate(choices)
        }
        trajectory_aliases = {
            _alias("trajectory", index): row
            for index, row in enumerate(pool.trajectories)
        }
        acquired_ids = tuple(
            pool.trajectories[0].belief.acquired_evidence if pool.trajectories else ()
        )
        if not acquired_ids:
            return TerminalAnswerDecision(
                status="abstain",
                selected_choice=None,
                rationale="no acquired real evidence",
            )
        observation_by_id = {
            str(node.node_id): node for node in real_observations
        }
        acquired_nodes = [
            observation_by_id.get(node_id, graph.node_by_id[node_id])
            for node_id in acquired_ids
        ]
        payload = {
            "question": pool.trajectories[0].belief.question,
            "choices": choice_aliases,
            "acquired_real_evidence": [
                _real_observation_payload(node)
                for node in acquired_nodes
            ],
            "final_trajectories": {
                alias: {
                    "hypothesis": row.hypothesis,
                    "status": row.status.value,
                    "missing_roles": list(row.belief.missing_roles),
                    "contradictions": list(row.belief.contradictions),
                    "grounded_role_evidence": [
                        {"role": role, "node_semantic_key": _node_key(graph, node_id)}
                        for role, node_id in row.belief.grounded_role_evidence
                    ],
                }
                for alias, row in trajectory_aliases.items()
            },
            "allowed_status": ["select", "abstain"],
            "required_output": {
                "only_keys": ["status", "choice", "rationale"],
                "choice": "one exact choice alias when select; null when abstain",
            },
            "contract": {
                "real_evidence_only": True,
                "hidden_answer_unavailable": True,
                "no_numeric_reward_score_probability_confidence_or_utility": True,
            },
        }
        for attempt in range(2):
            result = self.client.complete_json(
                task=(
                    "Select one answer only if acquired real evidence and the surviving "
                    "reasoning trajectories support it; otherwise abstain."
                    if attempt == 0
                    else "Repair the response to the exact categorical answer schema, "
                    "using one known alias or null and no numeric values."
                ),
                payload=payload,
            )
            try:
                if not isinstance(result, dict) or set(result) != {
                    "status",
                    "choice",
                    "rationale",
                }:
                    raise ValueError("terminal answer response schema is invalid")
                if _contains_number(result):
                    raise ValueError(
                        "terminal answer response contains a numeric value"
                    )
                status = str(result.get("status") or "")
                choice = result.get("choice")
                if status == "select":
                    if not isinstance(choice, str) or choice not in choice_aliases:
                        raise ValueError("terminal answer selected an unknown choice")
                    selected = choice_aliases[choice]
                elif status == "abstain":
                    if choice is not None:
                        raise ValueError("terminal abstention must use a null choice")
                    selected = None
                else:
                    raise ValueError("terminal answer status is invalid")
                return TerminalAnswerDecision(
                    status=status,
                    selected_choice=selected,
                    rationale=str(result.get("rationale") or ""),
                )
            except (TypeError, ValueError):
                if attempt == 1:
                    raise
        raise RuntimeError("unreachable terminal answer repair state")


class _FailClosedMultiTrajectoryPlanner:
    """Terminate without model calls when an upstream method contract fails."""

    def plan(
        self,
        pool: TrajectoryPool,
        graph: RetainedEvidenceGraph,
    ) -> MultiTrajectoryPlanDecision:
        trajectory = pool.expandable[0]
        action = next(
            row
            for row in GraphActionCompiler().compile(trajectory.belief, graph)
            if row.kind is ActionKind.ABSTAIN
        )
        expansion = TrajectoryExpansion(
            expansion_id=f"fail-closed:{trajectory.trajectory_id}",
            trajectory_id=trajectory.trajectory_id,
            action=action,
        )
        return MultiTrajectoryPlanDecision(
            selected_action=action,
            planning_status="abstain_entry_localization_failure",
            expansions=(expansion,),
            preferred_expansion_ids=(),
            lifecycle_predictions=(),
            legal_expansion_count=1,
        )


def run_multi_trajectory_case(
    *,
    case_id: str,
    question: str,
    choices: Sequence[str],
    graph: RetainedEvidenceGraph,
    clue_intervals: Sequence[Any],
    planner: Any,
    answer_selector: GPTOSSMultiTrajectoryAnswerSelector,
    read_budget: int,
    arm: str,
    initial_missing_roles: Sequence[str] = (),
    initial_entry_node_ids: Sequence[str] = (),
    belief_updater: Any | None = None,
    assessor: Any | None = None,
    evidence_reader: Any | None = None,
    evidence_sufficiency_evaluator: Any | None = None,
    evaluator_answer: str | None = None,
    max_decisions: int | None = None,
) -> dict[str, Any]:
    """Run first, then join hidden clue/answer labels for evaluation only."""

    if read_budget < 1:
        raise ValueError("read budget must be positive")
    belief = CursorBeliefState(
        belief_id=f"belief:{case_id}:{arm}:initial",
        question=question,
        localized_entry_node_ids=tuple(initial_entry_node_ids),
        required_roles=tuple(initial_missing_roles),
        missing_roles=tuple(initial_missing_roles),
        remaining_reads=read_budget,
    )
    pool = initialize_trajectory_pool(
        belief,
        tuple(str(choice) for choice in choices),
        pool_id=f"trajectory-pool:{case_id}:{arm}",
    )
    trace = run_multi_trajectory_closed_loop(
        pool,
        graph,
        planner,
        max_decisions=max_decisions or max(4, read_budget * 3 + 2),
        belief_updater=belief_updater,
        assessor=assessor,
        evidence_reader=evidence_reader,
    )
    # The selector has no access to evaluator_answer or clue intervals.
    answer = answer_selector.select(
        trace.final_pool,
        choices,
        graph,
        real_observations=trace.real_observations,
    )
    sufficiency: EvidenceSufficiencyDecision | None = None
    if evidence_sufficiency_evaluator is not None and evaluator_answer is not None:
        try:
            sufficiency = evidence_sufficiency_evaluator.evaluate(
                question=question,
                choices=choices,
                evaluator_answer=evaluator_answer,
                observations=trace.real_observations,
            )
        except Exception as exc:
            sufficiency = EvidenceSufficiencyDecision(
                "inconclusive",
                f"evaluator failure: {type(exc).__name__}: {exc}",
            )

    evaluator = HiddenClueCoverageEvaluator(clue_intervals)
    realized_rows: list[dict[str, Any]] = []
    for step in trace.steps:
        if step.observation_id is None:
            continue
        realized, outcome, newly_covered = evaluator.score_read(
            step.decision.selected_action,
            graph,
        )
        realized_rows.append(
            {
                "observation_id": step.observation_id,
                "observation_outcome": outcome.value,
                "belief_delta": _jsonable(realized),
                "newly_covered_clue_indices": list(newly_covered),
                "fed_back_to_planner": False,
            }
        )
    real_reads = len(realized_rows)
    first_read_gain = bool(
        realized_rows and realized_rows[0]["newly_covered_clue_indices"]
    )
    later_read_gain = any(
        row["newly_covered_clue_indices"] for row in realized_rows[1:]
    )
    clue_count = len(evaluator.clues)
    covered = len(evaluator.covered_indices)
    correct_trajectory = next(
        (
            row
            for row in trace.final_pool.trajectories
            if evaluator_answer is not None and row.hypothesis == evaluator_answer
        ),
        None,
    )
    correct_survived = (
        correct_trajectory is not None
        and correct_trajectory.status
        not in {
            TrajectoryStatus.CONTRADICTED,
            TrajectoryStatus.ABANDONED,
            TrajectoryStatus.MERGED,
        }
    )
    statuses = [row.status.value for row in trace.final_pool.trajectories]
    steps = [_jsonable(row) for row in trace.steps]
    empirical_audit = _multi_trajectory_empirical_audit(trace, realized_rows)
    return {
        "schema_version": "steam-multi-trajectory-closed-loop-run/v0.2",
        "runtime_contract_version": MULTI_RUNTIME_CONTRACT,
        "case_id": case_id,
        "arm": arm,
        "graph_fingerprint": graph_fingerprint(graph),
        "hypothesis_source": "public_answer_choices",
        "initial_hypothesis_count": len(choices),
        "steps": steps,
        "real_observations": [_jsonable(row) for row in trace.real_observations],
        "realized_labels_evaluator_only": realized_rows,
        "termination": trace.termination,
        "terminal_answer": _jsonable(answer),
        "evidence_sufficiency_evaluator_only": _jsonable(sufficiency),
        "final_trajectory_statuses": statuses,
        "empirical_audit": empirical_audit,
        "metrics": {
            "answer_correct": (
                answer.selected_choice == evaluator_answer
                if evaluator_answer is not None
                else None
            ),
            "answer_abstained": answer.selected_choice is None,
            "clue_count": clue_count,
            "covered_clue_count": covered,
            "clue_recall": covered / clue_count,
            "clue_coverage_complete": evaluator.complete,
            "real_read_count": real_reads,
            "read_efficiency": covered / real_reads if real_reads else 0.0,
            "delayed_reasoning_success": evaluator.complete
            and clue_count > 1
            and real_reads > 1,
            "delayed_recovery_success": (
                real_reads > 1 and not first_read_gain and later_read_gain
            ),
            "delayed_completion_after_unproductive_first": (
                evaluator.complete
                and real_reads > 1
                and not first_read_gain
                and later_read_gain
            ),
            "correct_hypothesis_survived": correct_survived,
            "false_correct_hypothesis_elimination": (
                correct_trajectory is not None and not correct_survived
            ),
            "final_expandable_trajectory_count": len(trace.final_pool.expandable),
            "shared_action_selection_count": sum(
                row.decision.planning_status.endswith("preferred_first_hop")
                or row.decision.planning_status
                == "selected_shared_action_across_trajectories"
                for row in trace.steps
            ),
            "complete_comparison_budget_failure": any(
                row.decision.planning_status
                == "rollout_abstain_complete_comparison_budget_exceeded"
                for row in trace.steps
            ),
            "joint_chain_outcome_retention_rate": empirical_audit[
                "joint_chain_coverage"
            ]["outcome_retention_rate"],
            "post_read_belief_divergence_rate": empirical_audit[
                "post_read_belief_divergence"
            ]["divergence_rate"],
            "transition_outcome_calibration_accuracy": empirical_audit[
                "transition_calibration"
            ]["observation_outcome_exact_accuracy"],
            "transition_progress_calibration_accuracy": empirical_audit[
                "transition_calibration"
            ]["progress_exact_accuracy"],
        },
        "hidden_evaluator_feedback_to_planner": False,
        "top_k_applied": False,
        "training_performed": False,
    }


def _multi_trajectory_empirical_audit(
    trace: Any,
    realized_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Audit coverage, correction divergence, and hidden-label calibration."""

    coverage_rows: list[dict[str, Any]] = []
    divergence_rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, str]] = []
    realized_by_observation = {str(row["observation_id"]): row for row in realized_rows}
    for step_index, step in enumerate(trace.steps):
        paths = tuple(step.decision.imagined_paths)
        preference_audit = step.decision.preference_audit or {}
        expected_outcomes = int(
            preference_audit.get(
                "hypothesis_conditioned_outcome_count",
                sum(len(path.conditioned_outcomes) for path in paths),
            )
        )
        retained_outcomes = sum(len(path.conditioned_outcomes) for path in paths)
        expandable_ids = {row.trajectory_id for row in step.pool_before.expandable}
        per_chain = [
            {
                "path_id": path.path_id,
                "conditioned_outcome_count": len(path.conditioned_outcomes),
                "covers_every_current_hypothesis": {
                    row.trajectory_id for row in path.conditioned_outcomes
                }
                == expandable_ids,
            }
            for path in paths
        ]
        coverage_rows.append(
            {
                "step_index": step_index,
                "joint_chain_count": len(paths),
                "expected_conditioned_outcome_count": expected_outcomes,
                "retained_conditioned_outcome_count": retained_outcomes,
                "lossless_joint_grouping": retained_outcomes == expected_outcomes,
                "every_chain_covers_every_current_hypothesis": all(
                    row["covers_every_current_hypothesis"] for row in per_chain
                ),
                "per_chain": per_chain,
            }
        )
        if step.observation_id is None:
            continue
        signatures: dict[tuple[Any, ...], list[str]] = defaultdict(list)
        for trajectory in step.pool_after.trajectories:
            belief = trajectory.belief
            signature = (
                belief.missing_roles,
                belief.grounded_role_evidence,
                belief.contradictions,
                belief.answerability.value,
            )
            signatures[signature].append(trajectory.hypothesis)
        divergence_rows.append(
            {
                "step_index": step_index,
                "observation_id": step.observation_id,
                "hypothesis_count": len(step.pool_after.trajectories),
                "unique_belief_signature_count": len(signatures),
                "belief_diverged": len(signatures) > 1,
                "belief_groups": [
                    {
                        "hypotheses": hypotheses,
                        "missing_roles": list(signature[0]),
                        "grounded_role_evidence": [list(row) for row in signature[1]],
                        "contradictions": list(signature[2]),
                        "answerability": signature[3],
                    }
                    for signature, hypotheses in signatures.items()
                ],
            }
        )
        realized = realized_by_observation[step.observation_id]
        selected_key = shared_action_key(step.decision.selected_action)
        for path in paths:
            if shared_action_key(path.first_action) != selected_key:
                continue
            for outcome in path.conditioned_outcomes:
                prediction = outcome.transitions[0]
                calibration_rows.append(
                    {
                        "predicted_observation_outcome": (
                            prediction.observation.outcome.value
                        ),
                        "realized_observation_outcome": str(
                            realized["observation_outcome"]
                        ),
                        "predicted_progress": (prediction.belief_delta.progress.value),
                        "realized_progress": str(realized["belief_delta"]["progress"]),
                        "predicted_answerability": (
                            prediction.belief_delta.answerability_after.value
                        ),
                        "realized_answerability": str(
                            realized["belief_delta"]["answerability_after"]
                        ),
                    }
                )
    retained = sum(row["retained_conditioned_outcome_count"] for row in coverage_rows)
    expected = sum(row["expected_conditioned_outcome_count"] for row in coverage_rows)
    return {
        "joint_chain_coverage": {
            "steps": coverage_rows,
            "all_steps_lossless": all(
                row["lossless_joint_grouping"] for row in coverage_rows
            ),
            "outcome_retention_rate": retained / expected if expected else None,
            "all_initial_chains_cover_every_hypothesis": (
                coverage_rows[0]["every_chain_covers_every_current_hypothesis"]
                if coverage_rows
                else None
            ),
        },
        "post_read_belief_divergence": {
            "steps": divergence_rows,
            "divergence_rate": (
                sum(row["belief_diverged"] for row in divergence_rows)
                / len(divergence_rows)
                if divergence_rows
                else None
            ),
        },
        "transition_calibration": {
            "reference": "hidden_clue_overlap_evaluator_only",
            "rows": calibration_rows,
            "observation_outcome_exact_accuracy": _categorical_accuracy(
                calibration_rows,
                "predicted_observation_outcome",
                "realized_observation_outcome",
            ),
            "progress_exact_accuracy": _categorical_accuracy(
                calibration_rows,
                "predicted_progress",
                "realized_progress",
            ),
            "answerability_exact_accuracy": _categorical_accuracy(
                calibration_rows,
                "predicted_answerability",
                "realized_answerability",
            ),
            "observation_outcome_confusion": _categorical_confusion(
                calibration_rows,
                "predicted_observation_outcome",
                "realized_observation_outcome",
            ),
        },
    }


def _categorical_accuracy(
    rows: Sequence[dict[str, str]],
    predicted_key: str,
    realized_key: str,
) -> float | None:
    if not rows:
        return None
    return sum(row[predicted_key] == row[realized_key] for row in rows) / len(rows)


def _categorical_confusion(
    rows: Sequence[dict[str, str]],
    predicted_key: str,
    realized_key: str,
) -> dict[str, dict[str, int]]:
    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in rows:
        confusion[row[realized_key]][row[predicted_key]] += 1
    return {
        realized: dict(sorted(predicted.items()))
        for realized, predicted in sorted(confusion.items())
    }


def run_multi_trajectory_matched_pilot(
    *,
    gate: dict[str, Any],
    dataset: dict[str, Any],
    hidden_key: dict[str, Any],
    graph_root: Path,
    client: Any,
    capacity: int,
    read_budget: int,
    case_limit: int = 8,
    case_ids: Iterable[str] = (),
    arms: Iterable[str] = MULTI_ARMS,
    transition_batch_size: int = 1,
    comparison_batch_size: int = 24,
    max_complete_pairs: int | None = 4096,
    include_caption_candidates: bool = True,
    question_role_cache_path: Path | None = None,
    cache_mode: str = "record",
    belief_backend: str = "latent",
    real_evidence_reader: Any | None = None,
    evidence_sufficiency_evaluator: Any | None = None,
) -> dict[str, Any]:
    if not gate.get("gate_passed"):
        raise ValueError("CG-Bench compile gate did not pass")
    if isinstance(
        getattr(evidence_sufficiency_evaluator, "client", None),
        PersistentCategoricalResponseCacheClient,
    ):
        raise ValueError(
            "hidden evidence-sufficiency evaluation cannot share the planner cache"
        )
    requested_arms = tuple(dict.fromkeys(arms))
    if set(requested_arms) - set(MULTI_ARMS):
        raise ValueError("unsupported multi-trajectory matched arm")
    if belief_backend not in {"latent", "gtsam_backup", "gtsam_always"}:
        raise ValueError("unknown multi-trajectory belief backend")
    public_by_case = {str(row["case_id"]): row for row in dataset.get("cases") or []}
    hidden_by_case = {str(row["case_id"]): row for row in hidden_key.get("cases") or []}
    requested_ids = set(case_ids)
    selected = [
        value
        for value in gate.get("runnable_case_ids") or []
        if not requested_ids or value in requested_ids
    ][:case_limit]
    if requested_ids - set(selected):
        raise ValueError("some requested cases are not runnable under the frozen gate")

    initializer: Any = GPTOSSQuestionBeliefInitializer(client)
    entry_localizer = GPTOSSEntryLocalizer(client)
    if question_role_cache_path is not None:
        initializer = PersistentQuestionRoleCache(
            initializer,
            question_role_cache_path,
            mode=cache_mode,
        )
    runs: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    method_failures: list[dict[str, str]] = []
    for case_id in selected:
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
        question = str(public["planner_input"]["question"])
        choices = tuple(
            str(row) for row in public["planner_input"].get("choices") or []
        )
        if not choices:
            raise ValueError(f"public choices are missing for {case_id}")
        roles: tuple[str, ...] = ()
        entry_node_ids: tuple[str, ...] = ()
        entry_failure: str | None = None
        try:
            roles = initializer.initialize(question)
            entry_node_ids = entry_localizer.localize(
                question=question,
                missing_roles=roles,
                graph=graph,
            )
        except Exception as exc:
            entry_failure = f"{type(exc).__name__}: {exc}"
            method_failures.append(
                {
                    "case_id": case_id,
                    "stage": "entry_localization",
                    "failure": entry_failure,
                }
            )
            entry_localizer.audits.append(
                {
                    "candidate_address_count": len(graph.nodes),
                    "selected_anchor_count": 0,
                    "selected_node_ids": [],
                    "status": "failed_contract",
                    "top_k_applied": False,
                    "numeric_score_used": False,
                    "failure": entry_failure,
                    "fail_closed": True,
                }
            )
        clues = _clues(hidden)
        for arm in requested_arms:
            try:
                if arm == "oracle_clue_ceiling":
                    run = run_oracle_clue_ceiling(
                        case_id=case_id,
                        question=question,
                        graph=graph,
                        clue_intervals=clues,
                        read_budget=read_budget,
                        initial_entry_node_ids=entry_node_ids,
                    )
                    run["oracle_scope"] = "matched_model_localized_entry_frontier"
                    run["metrics"]["answer_correct"] = None
                    run["metrics"]["correct_hypothesis_survived"] = None
                    run["metrics"]["false_correct_hypothesis_elimination"] = None
                else:
                    reread_audit_start = len(
                        getattr(real_evidence_reader, "audit_records", ())
                    )
                    model = GPTOSSCategoricalMultiTrajectoryModel(
                        client,
                        transition_batch_size=transition_batch_size,
                        comparison_batch_size=comparison_batch_size,
                    )
                    planner = (
                        _FailClosedMultiTrajectoryPlanner()
                        if entry_failure is not None
                        else _planner_for_arm(
                            arm,
                            model,
                            max_complete_pairs=max_complete_pairs,
                        )
                    )
                    updater: Any
                    if belief_backend == "latent":
                        updater = GPTOSSRealEvidenceBeliefUpdater(client)
                    else:
                        updater = GTSAMMultiTrajectoryBeliefUpdater(
                            loaded.overlay,
                            mode=(
                                "backup"
                                if belief_backend == "gtsam_backup"
                                else "correct"
                            ),
                        )
                    run = run_multi_trajectory_case(
                        case_id=case_id,
                        question=question,
                        choices=choices,
                        graph=graph,
                        clue_intervals=clues,
                        planner=planner,
                        answer_selector=GPTOSSMultiTrajectoryAnswerSelector(client),
                        read_budget=read_budget,
                        arm=arm,
                        initial_missing_roles=roles,
                        initial_entry_node_ids=entry_node_ids,
                        belief_updater=updater,
                        assessor=GPTOSSRealTrajectoryEvidenceAssessor(client),
                        evidence_reader=real_evidence_reader,
                        evidence_sufficiency_evaluator=(
                            evidence_sufficiency_evaluator
                        ),
                        evaluator_answer=str(hidden.get("answer_text") or ""),
                    )
                    run["method_audit"] = {
                        "planner_type": type(planner).__name__,
                        "world_model_type": type(
                            getattr(planner, "world_model", None)
                        ).__name__,
                        "complete_coverage": getattr(
                            planner, "last_complete_coverage_audit", {}
                        ),
                        "transport": model.transport_audits,
                        "top_k_applied": False,
                        "belief_backend": belief_backend,
                        "gtsam_audit": list(getattr(updater, "audit_records", ())),
                        "real_evidence_reread": list(
                            getattr(real_evidence_reader, "audit_records", ())
                        )[reread_audit_start:],
                        "entry_localization": entry_localizer.audits[-1],
                        "entry_localization_failure": entry_failure,
                    }
                if run["graph_fingerprint"] != expected_fingerprint:
                    raise ValueError("matched arm mutated the retained graph")
                runs.append(run)
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
                    fallback = run_multi_trajectory_case(
                        case_id=case_id,
                        question=question,
                        choices=choices,
                        graph=graph,
                        clue_intervals=clues,
                        planner=_FailClosedMultiTrajectoryPlanner(),
                        answer_selector=GPTOSSMultiTrajectoryAnswerSelector(client),
                        read_budget=read_budget,
                        arm=arm,
                        initial_missing_roles=roles,
                        initial_entry_node_ids=entry_node_ids,
                        evaluator_answer=str(hidden.get("answer_text") or ""),
                    )
                    fallback["method_audit"] = {
                        "planner_type": "FailClosedMultiTrajectoryPlanner",
                        "arm_runtime_failed": True,
                        "failure": failure,
                        "entry_localization": entry_localizer.audits[-1],
                        "top_k_applied": False,
                    }
                    runs.append(fallback)

    divergences: list[dict[str, Any]] = []
    by_case: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for run in runs:
        by_case[str(run["case_id"])][str(run["arm"])] = run
    for case_id, case_runs in by_case.items():
        reference = case_runs.get("world_model_guided")
        if reference is None:
            continue
        for arm, candidate in case_runs.items():
            if arm != "world_model_guided":
                divergences.append(
                    {"case_id": case_id, **action_divergence(reference, candidate)}
                )
    localized_entry_preflight = []
    scientifically_evaluable_case_ids: list[str] = []
    for case_id in selected:
        oracle = by_case.get(case_id, {}).get("oracle_clue_ceiling")
        complete = bool(
            oracle and oracle.get("metrics", {}).get("clue_coverage_complete")
        )
        if complete:
            scientifically_evaluable_case_ids.append(case_id)
        localized_entry_preflight.append(
            {
                "case_id": case_id,
                "status": "complete" if complete else "insufficient_frontier",
                "clue_recall_ceiling": (
                    oracle.get("metrics", {}).get("clue_recall") if oracle else None
                ),
                "clue_coverage_complete": complete,
                "evaluator_only": True,
                "fed_back_to_planner": False,
            }
        )
    evaluable_runs = [
        run
        for run in runs
        if str(run["case_id"]) in set(scientifically_evaluable_case_ids)
    ]
    return {
        "schema_version": MULTI_PILOT_SCHEMA,
        "runtime_contract_version": MULTI_RUNTIME_CONTRACT,
        "dataset_id": dataset.get("dataset_id"),
        "model": str(getattr(client, "model", "unknown")),
        "arms": list(requested_arms),
        "selected_case_ids": selected,
        "matched_contract": {
            "same_frozen_graph": True,
            "same_public_choice_hypotheses": True,
            "same_real_read_budget": read_budget,
            "hidden_feedback_to_planner": False,
            "model_output_is_categorical_only": True,
            "top_k_applied": False,
            "complete_pair_budget": max_complete_pairs,
            "budget_overflow_policy": "explicit_abstain_not_candidate_pruning",
            "belief_backend": belief_backend,
            "gtsam_is_optional_and_never_ranks_actions": True,
            "same_entry_localization_across_model_backed_arms": True,
            "hidden_evidence_sufficiency_evaluator_is_uncached_and_posthoc": True,
        },
        "runs": runs,
        "errors": errors,
        "method_failures": method_failures,
        "metrics_by_arm": _aggregate(runs, requested_arms),
        "metrics_by_arm_localized_frontier_complete": _aggregate(
            evaluable_runs, requested_arms
        ),
        "localized_entry_preflight": localized_entry_preflight,
        "scientifically_evaluable_case_ids": scientifically_evaluable_case_ids,
        "localized_entry_preflight_contract": {
            "uses_hidden_clues_for_evaluation_only": True,
            "runs_after_model_entry_localization": True,
            "same_read_budget_as_model_arms": True,
            "fed_back_to_planner": False,
            "purpose": "separate navigation quality from unreachable evidence",
        },
        "action_divergence": divergences,
        "training_performed": False,
    }


def _planner_for_arm(
    arm: str,
    model: GPTOSSCategoricalMultiTrajectoryModel,
    *,
    max_complete_pairs: int | None,
) -> Any:
    if arm == "no_world_model":
        return ReactiveMultiTrajectoryPlanner(
            model,
            setwise_preference_model=model,
            evidence_scheduler=model,
        )
    world_model: Any = model
    horizon = 2
    if arm == "shuffled_world_model_prediction":
        world_model = ShuffledHypothesisWorldModel(model)
    elif arm == "immediate_effect_only":
        horizon = 1
    elif arm != "world_model_guided":
        raise ValueError(f"unsupported model-backed arm: {arm}")
    return MultiTrajectoryRolloutPlanner(
        world_model,
        model,
        horizon=horizon,
        max_complete_pairs=max_complete_pairs,
        setwise_preference_model=model,
        evidence_scheduler=model,
    )


def _aggregate(runs: Sequence[dict[str, Any]], arms: Iterable[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for arm in arms:
        metrics = [row["metrics"] for row in runs if row["arm"] == arm]
        result[arm] = {
            "case_count": len(metrics),
            "answer_accuracy": _mean(row.get("answer_correct") for row in metrics),
            "mean_clue_recall": _mean(row.get("clue_recall") for row in metrics),
            "mean_real_reads": _mean(row.get("real_read_count") for row in metrics),
            "mean_read_efficiency": _mean(
                row.get("read_efficiency") for row in metrics
            ),
            "answer_abstain_rate": _mean(
                row.get("answer_abstained") for row in metrics
            ),
            "delayed_success_rate": _mean(
                row.get("delayed_reasoning_success") for row in metrics
            ),
            "delayed_recovery_rate": _mean(
                row.get("delayed_recovery_success") for row in metrics
            ),
            "delayed_completion_after_unproductive_first_rate": _mean(
                row.get("delayed_completion_after_unproductive_first")
                for row in metrics
            ),
            "correct_hypothesis_survival_rate": _mean(
                row.get("correct_hypothesis_survived") for row in metrics
            ),
            "false_correct_hypothesis_elimination_rate": _mean(
                row.get("false_correct_hypothesis_elimination") for row in metrics
            ),
            "complete_comparison_budget_failure_rate": _mean(
                row.get("complete_comparison_budget_failure") for row in metrics
            ),
            "mean_joint_chain_outcome_retention": _mean(
                row.get("joint_chain_outcome_retention_rate") for row in metrics
            ),
            "post_read_belief_divergence_rate": _mean(
                row.get("post_read_belief_divergence_rate") for row in metrics
            ),
            "transition_outcome_calibration_accuracy": _mean(
                row.get("transition_outcome_calibration_accuracy") for row in metrics
            ),
            "transition_progress_calibration_accuracy": _mean(
                row.get("transition_progress_calibration_accuracy") for row in metrics
            ),
        }
    return result


def _mean(values: Iterable[Any]) -> float | None:
    selected = [float(row) for row in values if row is not None]
    return fmean(selected) if selected else None


def _node_key(graph: RetainedEvidenceGraph, node_id: str) -> str:
    node = graph.node_by_id.get(node_id)
    return (
        str(node.metadata.get("predicate") or node.text or node_id)
        if node
        else "unknown"
    )


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {key: _jsonable(row) for key, row in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): _jsonable(row) for key, row in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(row) for row in value]
    return value


def _real_observation_payload(node: Any) -> dict[str, Any]:
    """Preserve grounded L1/reread structure instead of collapsing to a caption."""

    metadata = node.metadata or {}
    structured_keys = (
        "action_kind",
        "participants",
        "states",
        "state_change",
        "actor",
        "target",
        "objects",
        "location",
        "visual_reread",
        "reread_descriptor",
    )
    return {
        "node_id": str(node.node_id),
        "semantic_key": str(metadata.get("predicate") or node.text or node.node_id),
        "evidence_value": str(node.text or metadata.get("predicate") or ""),
        "time_span": {
            "start_s": node.time_span.start_s,
            "end_s": node.time_span.end_s,
        },
        "structured_observation": {
            key: _jsonable(metadata[key])
            for key in structured_keys
            if metadata.get(key) not in (None, "", [], {})
        },
        "provenance": {
            "video_id": str(node.video_id),
            "source_segments": list(node.source_segments),
            "producer": str(node.provenance.get("producer") or ""),
        },
    }


def _contains_number(value: Any) -> bool:
    if isinstance(value, bool) or value is None:
        return False
    if isinstance(value, (int, float)):
        return True
    if isinstance(value, dict):
        return any(_contains_number(row) for row in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_number(row) for row in value)
    return False


def _alias(prefix: str, index: int) -> str:
    value = index
    letters = ""
    while True:
        letters = chr(ord("a") + value % 26) + letters
        value = value // 26 - 1
        if value < 0:
            break
    return f"{prefix}_{letters}"


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
        "--compiled-gate",
        type=Path,
        help="Reuse a previously frozen compile-gate artifact instead of recompiling.",
    )
    parser.add_argument("--keys-py", type=Path)
    parser.add_argument(
        "--mode", choices=("compile-gate", "run"), default="compile-gate"
    )
    parser.add_argument("--model", default=DEFAULT_GPT_OSS_MODEL)
    parser.add_argument("--capacity", type=int, default=64)
    parser.add_argument("--graph-read-budget", type=int, default=8)
    parser.add_argument("--video-limit", type=int, default=8)
    parser.add_argument("--cases-per-video", type=int, default=1)
    parser.add_argument("--case-limit", type=int, default=8)
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--arm", action="append", choices=MULTI_ARMS)
    parser.add_argument(
        "--transition-batch-size",
        type=int,
        default=1,
        help="Number of compact shared-action groups per model request.",
    )
    parser.add_argument("--comparison-batch-size", type=int, default=24)
    parser.add_argument("--max-complete-pairs", type=int, default=4096)
    parser.add_argument("--question-role-cache", type=Path)
    parser.add_argument("--response-cache", type=Path)
    parser.add_argument("--cache-mode", choices=("record", "replay"), default="record")
    parser.add_argument("--timeout-s", type=int, default=180)
    parser.add_argument("--max-tokens", type=int, default=8000)
    parser.add_argument(
        "--reasoning-effort", choices=("low", "medium", "high"), default="low"
    )
    parser.add_argument("--disable-caption-candidates", action="store_true")
    parser.add_argument(
        "--belief-backend",
        choices=("latent", "gtsam_backup", "gtsam_always"),
        default="latent",
    )
    parser.add_argument(
        "--real-evidence-reread-artifact",
        type=Path,
        help=(
            "Optional human/ground-truth-locked raw-clip reread artifact; applied "
            "only after a legal real read."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    dataset = _read_json(args.dataset)
    hidden = _read_json(args.hidden_key)
    selection = _read_json(args.selection)
    gate = (
        _read_json(args.compiled_gate)
        if args.compiled_gate is not None
        else compile_cgbench_gate(
            dataset=dataset,
            hidden_key=hidden,
            selection=selection,
            graph_root=args.graph_root,
            capacity=args.capacity,
            read_budget=args.graph_read_budget,
            video_limit=args.video_limit,
            cases_per_video=args.cases_per_video,
            include_caption_candidates=not args.disable_caption_candidates,
            allowed_case_ids=args.case_id,
        )
    )
    if args.compiled_gate is not None:
        if gate.get("dataset_id") != dataset.get("dataset_id"):
            raise ValueError("compiled gate dataset does not match --dataset")
        if int(gate.get("capacity", -1)) != args.capacity:
            raise ValueError("compiled gate capacity does not match --capacity")
        if int(gate.get("read_budget", -1)) != args.graph_read_budget:
            raise ValueError(
                "compiled gate read budget does not match --graph-read-budget"
            )
        if Path(str(gate.get("graph_root") or "")).resolve() != (
            args.graph_root.expanduser().resolve()
        ):
            raise ValueError("compiled gate graph root does not match --graph-root")
    if args.mode == "compile-gate":
        _write_json(args.output, gate)
        return 0 if gate["gate_passed"] else 2
    if args.keys_py is None:
        raise ValueError("--mode run requires --keys-py")
    uncached_client: Any = OpenAICompatibleCategoricalClient.from_openrouter_keys_file(
        args.keys_py,
        model=args.model,
        timeout_s=args.timeout_s,
        max_tokens=args.max_tokens,
        reasoning_effort=args.reasoning_effort,
    )
    client: Any = uncached_client
    if args.response_cache is not None:
        client = PersistentCategoricalResponseCacheClient(
            client,
            args.response_cache,
            mode=args.cache_mode,
        )
    result = run_multi_trajectory_matched_pilot(
        gate=gate,
        dataset=dataset,
        hidden_key=hidden,
        graph_root=args.graph_root,
        client=client,
        capacity=args.capacity,
        read_budget=args.graph_read_budget,
        case_limit=args.case_limit,
        case_ids=args.case_id,
        arms=args.arm or MULTI_ARMS,
        transition_batch_size=args.transition_batch_size,
        comparison_batch_size=args.comparison_batch_size,
        max_complete_pairs=args.max_complete_pairs,
        include_caption_candidates=not args.disable_caption_candidates,
        question_role_cache_path=args.question_role_cache,
        cache_mode=args.cache_mode,
        belief_backend=args.belief_backend,
        evidence_sufficiency_evaluator=GPTOSSEvidenceSufficiencyEvaluator(
            uncached_client
        ),
        real_evidence_reader=(
            ArtifactBackedRealEvidenceReader(args.real_evidence_reread_artifact)
            if args.real_evidence_reread_artifact is not None
            else None
        ),
    )
    result["compile_gate"] = gate
    _write_json(args.output, result)
    print(json.dumps(result["metrics_by_arm"], indent=2))
    return 0 if not result["errors"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
