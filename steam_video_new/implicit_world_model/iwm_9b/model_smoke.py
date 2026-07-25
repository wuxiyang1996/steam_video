"""Small real-model smoke for the structured multi-path IWM/Planner loop."""

from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass, replace
from enum import Enum
import json
from pathlib import Path
from typing import Any

from memory_graph.types import MemoryNode, TimeSpan

from ..full_graph_iwm.contracts import (
    CursorBeliefState,
    RetainedEvidenceGraph,
    TemporalNavigationEdge,
)
from ..full_graph_iwm.multi_trajectory import (
    initialize_trajectory_pool,
    run_multi_trajectory_closed_loop,
)
from ..full_graph_iwm.multi_trajectory_rollout import MultiTrajectoryRolloutPlanner
from ..full_graph_iwm.transition_cache import (
    PersistentCategoricalResponseCacheClient,
)
from ..l15_graph_navigator.gpt_oss import OpenAICompatibleCategoricalClient
from .runtime import Structured9BMultiTrajectoryModel


def run_model_smoke(
    *,
    client: Any,
    output: Path,
) -> dict[str, Any]:
    graph = _smoke_graph()
    pool = initialize_trajectory_pool(
        CursorBeliefState(
            belief_id="belief:gpt5mini-smoke:initial",
            question="Who opened the red cabinet after entering the workshop?",
            localized_entry_node_ids=("l1:entry",),
            required_roles=("entrant_identity", "later_cabinet_action"),
            missing_roles=("entrant_identity", "later_cabinet_action"),
            remaining_reads=2,
        ),
        (
            "the orange-haired entrant opened the red cabinet",
            "another person opened the red cabinet",
        ),
    )
    model = Structured9BMultiTrajectoryModel(
        client,
        transition_batch_size=16,
        comparison_batch_size=16,
    )
    planner = MultiTrajectoryRolloutPlanner(
        model,
        model,
        horizon=2,
        setwise_preference_model=model,
    )
    trace = run_multi_trajectory_closed_loop(
        pool,
        graph,
        planner,
        max_decisions=2,
        belief_updater=_GroundedSmokeBeliefUpdater(),
    )
    coverage = [
        {
            "step": index,
            "active_hypothesis_ids": sorted(
                row.trajectory_id for row in step.pool_before.expandable
            ),
            "joint_tree_count": len(step.decision.imagined_paths),
            "every_tree_covers_every_active_hypothesis": all(
                {outcome.trajectory_id for outcome in tree.conditioned_outcomes}
                == {row.trajectory_id for row in step.pool_before.expandable}
                for tree in step.decision.imagined_paths
            ),
            "rich_patch_present": all(
                bool(transition.structured_patch)
                for tree in step.decision.imagined_paths
                for outcome in tree.conditioned_outcomes
                for transition in outcome.transitions
            ),
        }
        for index, step in enumerate(trace.steps)
    ]
    real_reads = [
        step.observation_id for step in trace.steps if step.observation_id is not None
    ]
    result = {
        "schema_version": "steam-iwm-9b-real-model-smoke/v0.1",
        "model": str(getattr(client, "model", "unknown")),
        "method": {
            "iwm": "structured_belief_event_patch",
            "planner": "complete_shared_first_hop_tree_preference",
            "horizon": "two",
            "persistent_multi_path": True,
            "belief_correction": "executed_grounded_fixture_only",
            "numeric_reward_present": False,
            "top_k_applied": False,
        },
        "trace": _jsonable(trace),
        "runtime_audits": model.audits,
        "coverage": coverage,
        "gate": {
            "two_real_reads_completed": len(real_reads) == 2,
            "real_read_node_ids": real_reads,
            "all_joint_trees_complete": bool(coverage)
            and all(
                row["every_tree_covers_every_active_hypothesis"] for row in coverage
            ),
            "all_imagined_transitions_have_rich_patch": bool(coverage)
            and all(row["rich_patch_present"] for row in coverage),
            "no_imagined_evidence_in_persistent_belief": all(
                not row.belief.imagined_evidence
                for row in trace.final_pool.trajectories
            ),
            "multiple_reasoning_paths_retained": len(trace.final_pool.trajectories)
            == len(pool.trajectories),
        },
        "training_performed": False,
        "scientific_validation": False,
    }
    result["gate"]["passed"] = all(
        value
        for key, value in result["gate"].items()
        if key not in {"real_read_node_ids", "passed"}
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return result


class _GroundedSmokeBeliefUpdater:
    """Update only the role directly grounded by the executed smoke node."""

    def update_for_trajectory(
        self,
        trajectory_id: str,
        hypothesis: str,
        previous_belief: CursorBeliefState,
        structurally_updated_belief: CursorBeliefState,
        action: Any,
        observation: MemoryNode,
        graph: RetainedEvidenceGraph,
    ) -> CursorBeliefState:
        del trajectory_id, previous_belief, action, graph
        role = {
            "l1:entry": "entrant_identity",
            "l1:cabinet": "later_cabinet_action",
        }.get(observation.node_id)
        if role is None or role not in structurally_updated_belief.missing_roles:
            return structurally_updated_belief
        missing = tuple(
            value
            for value in structurally_updated_belief.missing_roles
            if value != role
        )
        bindings = dict(structurally_updated_belief.grounded_role_evidence)
        bindings[role] = observation.node_id
        contradictions = structurally_updated_belief.contradictions
        if observation.node_id == "l1:entry" and hypothesis.startswith(
            "another person"
        ):
            contradictions = (*contradictions, "entry_identity_counterevidence")
        return replace(
            structurally_updated_belief,
            belief_id=f"{structurally_updated_belief.belief_id}:grounded-correction",
            missing_roles=missing,
            grounded_role_evidence=tuple(
                (name, bindings[name])
                for name in structurally_updated_belief.required_roles
                if name in bindings
            ),
            contradictions=contradictions,
        )


def _smoke_graph() -> RetainedEvidenceGraph:
    entry = MemoryNode(
        node_id="l1:entry",
        video_id="video:gpt5mini-smoke",
        time_span=TimeSpan(0.0, 1.0),
        provenance={"producer": "grounded-smoke-fixture"},
        node_type="observation",
        text="An orange-haired person enters the workshop.",
        metadata={
            "predicate": "orange-haired person enters workshop",
            "layer": "L1",
        },
    )
    cabinet = MemoryNode(
        node_id="l1:cabinet",
        video_id="video:gpt5mini-smoke",
        time_span=TimeSpan(2.0, 3.0),
        provenance={"producer": "grounded-smoke-fixture"},
        node_type="observation",
        text="The same orange-haired person later opens the red cabinet.",
        metadata={
            "predicate": "same person later opens red cabinet",
            "layer": "L1",
        },
    )
    return RetainedEvidenceGraph(
        graph_id="graph:gpt5mini-structured-smoke",
        nodes=(entry, cabinet),
        temporal_edges=(
            TemporalNavigationEdge(
                edge_id="temporal:entry-cabinet",
                src=entry.node_id,
                dst=cabinet.node_id,
                relation="temporal_next",
            ),
        ),
        correlation_edges=(),
        capacity=2,
    )


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {key: _jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--keys-py", type=Path, required=True)
    parser.add_argument("--model", default="openai/gpt-5-mini")
    parser.add_argument("--api-base", default="https://openrouter.ai/api/v1")
    parser.add_argument("--timeout-s", type=int, default=180)
    parser.add_argument("--max-tokens", type=int, default=12000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    client = OpenAICompatibleCategoricalClient.from_openrouter_keys_file(
        args.keys_py,
        api_base=args.api_base,
        model=args.model,
        timeout_s=args.timeout_s,
        max_tokens=args.max_tokens,
        reasoning_effort="low",
    )
    client = PersistentCategoricalResponseCacheClient(
        client,
        args.output.with_suffix(args.output.suffix + ".responses.json"),
        mode="record",
    )
    result = run_model_smoke(client=client, output=args.output)
    print(json.dumps(result["gate"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
