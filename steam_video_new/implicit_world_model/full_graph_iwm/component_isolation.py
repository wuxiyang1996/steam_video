"""Teacher-forced component isolation for multi-trajectory IWM planning.

The diagnostic composes observation prediction and hypothesis-effect prediction
without changing the Planner.  It deliberately refuses to call dataset clue
overlap an oracle hypothesis effect.
"""

from __future__ import annotations

from dataclasses import replace
import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .contracts import ImaginedTransition, RetainedEvidenceGraph
from .multi_trajectory_rollout import (
    CategoricalHypothesisPathPreference,
    CategoricalHypothesisWorldModel,
    HypothesisExpansionRequest,
    MultiTrajectoryRolloutPlanner,
    ReactiveMultiTrajectoryPlanner,
)


ISOLATION_ARMS = (
    "oracle_observation_oracle_effect",
    "predicted_observation_oracle_effect",
    "predicted_observation_predicted_effect",
    "no_world_model",
)


class ComposedTeacherForcedWorldModel:
    """Compose observation and belief-effect channels from two world models."""

    def __init__(
        self,
        predicted_model: CategoricalHypothesisWorldModel,
        oracle_model: CategoricalHypothesisWorldModel,
        *,
        observation_source: str,
        effect_source: str,
    ) -> None:
        if observation_source not in {"predicted", "oracle"}:
            raise ValueError("unsupported observation source")
        if effect_source not in {"predicted", "oracle"}:
            raise ValueError("unsupported effect source")
        self.predicted_model = predicted_model
        self.oracle_model = oracle_model
        self.observation_source = observation_source
        self.effect_source = effect_source
        self.model_name = (
            f"teacher-forced:observation={observation_source}:effect={effect_source}"
        )

    def predict_batch(
        self,
        requests: Sequence[HypothesisExpansionRequest],
        graph: RetainedEvidenceGraph,
    ) -> Sequence[ImaginedTransition]:
        predicted = tuple(self.predicted_model.predict_batch(requests, graph))
        oracle = tuple(self.oracle_model.predict_batch(requests, graph))
        if len(predicted) != len(requests) or len(oracle) != len(requests):
            raise ValueError("teacher-forced transition coverage mismatch")
        composed: list[ImaginedTransition] = []
        for request, predicted_row, oracle_row in zip(requests, predicted, oracle):
            if (
                predicted_row.action != request.action
                or oracle_row.action != request.action
            ):
                raise ValueError("teacher-forced source changed the legal action")
            observation_row = (
                predicted_row if self.observation_source == "predicted" else oracle_row
            )
            effect_row = (
                predicted_row if self.effect_source == "predicted" else oracle_row
            )
            composed.append(
                replace(
                    effect_row,
                    observation=observation_row.observation,
                    action=request.action,
                )
            )
        return tuple(composed)


def build_component_isolation_planners(
    *,
    predicted_model: CategoricalHypothesisWorldModel,
    oracle_model: CategoricalHypothesisWorldModel,
    preference_model: CategoricalHypothesisPathPreference,
    horizon: int = 2,
    setwise_preference_model: Any | None = None,
    evidence_scheduler: Any | None = None,
) -> dict[str, Any]:
    """Build four planners that differ only at the intended component boundary."""

    common = {
        "horizon": horizon,
        "setwise_preference_model": setwise_preference_model,
        "evidence_scheduler": evidence_scheduler,
    }
    return {
        "oracle_observation_oracle_effect": MultiTrajectoryRolloutPlanner(
            ComposedTeacherForcedWorldModel(
                predicted_model,
                oracle_model,
                observation_source="oracle",
                effect_source="oracle",
            ),
            preference_model,
            **common,
        ),
        "predicted_observation_oracle_effect": MultiTrajectoryRolloutPlanner(
            ComposedTeacherForcedWorldModel(
                predicted_model,
                oracle_model,
                observation_source="predicted",
                effect_source="oracle",
            ),
            preference_model,
            **common,
        ),
        "predicted_observation_predicted_effect": MultiTrajectoryRolloutPlanner(
            predicted_model,
            preference_model,
            **common,
        ),
        "no_world_model": ReactiveMultiTrajectoryPlanner(
            preference_model,
            setwise_preference_model=setwise_preference_model,
            evidence_scheduler=evidence_scheduler,
        ),
    }


def component_isolation_readiness(
    effect_records: Sequence[Mapping[str, Any]],
    planner_records: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Fail closed until real hypothesis effects and preferences are available."""

    independently_supervised = [
        row
        for row in effect_records
        if row.get("target", {}).get("hypothesis_effect", {}).get("supervised") is True
    ]
    effect_groups: dict[str, set[str]] = {}
    for row in independently_supervised:
        observation_id = str(row.get("input", {}).get("observation_record_id") or "")
        value = row.get("target", {}).get("hypothesis_effect", {}).get("value")
        effect_groups.setdefault(observation_id, set()).add(_stable(value))
    divergent_groups = sum(len(values) > 1 for values in effect_groups.values())
    supervised_preferences = [
        row
        for row in planner_records
        if row.get("target", {}).get("preference", {}).get("supervised") is True
        and row.get("training_eligible") is True
    ]
    checks = {
        "independent_hypothesis_effects_present": bool(independently_supervised),
        "hypothesis_effect_divergence_present": divergent_groups > 0,
        "grounded_planner_preferences_present": bool(supervised_preferences),
    }
    return {
        "schema_version": "steam-iwm-component-isolation-readiness/v0.1",
        "required_arms": list(ISOLATION_ARMS),
        "independently_supervised_effect_count": len(independently_supervised),
        "effect_group_count": len(effect_groups),
        "divergent_effect_group_count": divergent_groups,
        "grounded_planner_preference_count": len(supervised_preferences),
        "checks": checks,
        "passed": all(checks.values()),
        "blockers": [name for name, passed in checks.items() if not passed],
        "dataset_clue_overlap_used_as_hypothesis_effect": False,
        "training_performed": False,
    }


def _stable(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _read_jsonl(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--effect-records", required=True, type=Path)
    parser.add_argument("--planner-records", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    report = component_isolation_readiness(
        _read_jsonl(args.effect_records),
        _read_jsonl(args.planner_records),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
