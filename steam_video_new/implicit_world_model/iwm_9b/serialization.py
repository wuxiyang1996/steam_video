"""Stable chat serialization shared by dry-runs and LoRA training."""

from __future__ import annotations

import json
from typing import Any, Mapping

from .schemas import (
    HYPOTHESIS_EFFECT_TASK,
    OBSERVATION_TASK,
    PLANNER_TASK,
    TRANSITION_TASK,
    supervised_target,
    validate_sft_record,
)


SYSTEM_PROMPTS = {
    TRANSITION_TASK: (
        "You are an action-conditioned implicit world model. Predict only the "
        "supervised structured future fields. Do not emit reward, utility, "
        "probability, confidence, Q-values, or unseen evidence as fact."
    ),
    OBSERVATION_TASK: (
        "You are an action-conditioned observation world model. Given the "
        "grounded L1/L1.5 state available before a legal read, predict only "
        "the compact future observation descriptor. Do not predict dataset "
        "clue labels, reward, utility, probability, confidence, or Q-values."
    ),
    HYPOTHESIS_EFFECT_TASK: (
        "You are a hypothesis-conditioned belief transition model. Predict "
        "only independently supervised effects of grounded evidence on the "
        "given hypothesis. Abstain when the effect is not established. Do not "
        "emit reward, utility, probability, confidence, or Q-values."
    ),
    PLANNER_TASK: (
        "You are a multi-path reasoning planner. Compare both complete available "
        "joint reasoning candidates. Return only prefer_left, prefer_right, tie, "
        "or incomparable. Do not emit a numeric score."
    ),
}


def render_sft_example(record: Mapping[str, Any]) -> dict[str, Any]:
    validate_sft_record(record)
    task = str(record["task"])
    assistant_target = supervised_target(record["target"])
    return {
        "record_id": record["record_id"],
        "task": task,
        "split": record["split"],
        "training_eligible": record["training_eligible"],
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPTS[task]},
            {
                "role": "user",
                "content": _json(record["input"]),
            },
            {
                "role": "assistant",
                "content": _json(assistant_target),
            },
        ],
    }


def _json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
