"""Evaluate a categorical proxy on blinded grounded-survivor pairs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from steam_video_new.implicit_world_model.l15_graph_navigator.gpt_oss import (
    OpenAICompatibleCategoricalClient,
)

from .transition_cache import PersistentCategoricalResponseCacheClient


SCHEMA = "steam-full-iwm-grounded-survivor-preference-eval/v0.1"
LABELS = {"prefer_left", "prefer_right", "tie", "incomparable"}


def evaluate_blinded_preferences(
    *,
    packet: dict[str, Any],
    hidden_key: dict[str, Any],
    client: Any,
    batch_size: int = 16,
    swap_sides: bool = False,
) -> dict[str, Any]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    hidden_by_id = {
        str(row["pair_id"]): row for row in hidden_key.get("records") or []
    }
    original_records = tuple(packet.get("records") or [])
    records = tuple(
        {
            **row,
            "left": row["right"],
            "right": row["left"],
        }
        if swap_sides
        else row
        for row in original_records
    )
    if {str(row["pair_id"]) for row in records} != set(hidden_by_id):
        raise ValueError("blinded packet and hidden key pair coverage differs")
    predictions: list[dict[str, Any]] = []
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        aliases = {
            str(row["pair_id"]): _alphabetic_alias(index)
            for index, row in enumerate(batch)
        }
        payload = {
            "independent_pairs": {
                aliases[str(row["pair_id"])]: {
                    "belief": row["belief"],
                    "left": row["left"],
                    "right": row["right"],
                }
                for row in batch
            },
            "allowed_labels": sorted(LABELS),
            "required_output": {
                "only_keys": ["decisions"],
                "decisions": {
                    alias: {
                        "label": "one allowed categorical label",
                        "rationale": "short categorical rationale",
                    }
                    for alias in aliases.values()
                },
            },
            "required_contract": {
                "pairs_are_independent": True,
                "every_pair_must_be_judged": True,
                "no_numeric_reward_score_probability_confidence_or_utility": True,
            },
        }
        result = client.complete_json(
            task=(
                "Independently compare each pair of imagined reasoning trajectories. "
                "Choose a strict side only when its predicted consequence is more useful "
                "for resolving the supplied belief; otherwise return tie or incomparable."
            ),
            payload=payload,
        )
        if set(result) != {"decisions"} or _contains_number(result):
            raise ValueError("proxy output must be categorical decisions only")
        decisions = result.get("decisions")
        if not isinstance(decisions, dict) or set(decisions) != set(aliases.values()):
            raise ValueError("proxy decisions do not exactly cover blinded pairs")
        pair_by_alias = {alias: pair_id for pair_id, alias in aliases.items()}
        for alias, decision in decisions.items():
            if not isinstance(decision, dict) or set(decision) != {"label", "rationale"}:
                raise ValueError("proxy decision fields do not match schema")
            label = str(decision.get("label") or "")
            if label not in LABELS:
                raise ValueError("proxy emitted an invalid categorical label")
            predictions.append(
                {
                    "pair_id": pair_by_alias[alias],
                    "prediction": label,
                    "rationale": str(decision.get("rationale") or ""),
                }
            )
    confusion: dict[str, dict[str, int]] = {}
    correct = 0
    for row in predictions:
        original_target = str(hidden_by_id[row["pair_id"]]["label"])
        target = _swapped_label(original_target) if swap_sides else original_target
        predicted = str(row["prediction"])
        confusion.setdefault(target, {})[predicted] = (
            confusion.setdefault(target, {}).get(predicted, 0) + 1
        )
        correct += int(target == predicted)
        row["target"] = target
        row["correct"] = target == predicted
    count = len(predictions)
    prediction_counts: dict[str, int] = {}
    for row in predictions:
        label = str(row["prediction"])
        prediction_counts[label] = prediction_counts.get(label, 0) + 1
    return {
        "schema_version": SCHEMA,
        "model": str(client.model),
        "pair_count": count,
        "correct_count": correct,
        "accuracy": (correct / count) if count else None,
        "confusion_matrix": confusion,
        "prediction_counts": prediction_counts,
        "predictions": predictions,
        "side_swap_applied": swap_sides,
        "input_was_blinded": True,
        "hidden_labels_sent_to_model": False,
        "numeric_reward_present": False,
        "training_performed": False,
    }


def _swapped_label(label: str) -> str:
    if label == "prefer_left":
        return "prefer_right"
    if label == "prefer_right":
        return "prefer_left"
    return label


def _alphabetic_alias(index: int) -> str:
    value = index
    suffix = ""
    while True:
        value, remainder = divmod(value, 26)
        suffix = chr(ord("a") + remainder) + suffix
        if value == 0:
            return "pair_" + suffix
        value -= 1


def _contains_number(value: Any) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return False
    if isinstance(value, (int, float)):
        return True
    if isinstance(value, dict):
        return any(_contains_number(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_number(item) for item in value)
    return False


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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", required=True, type=Path)
    parser.add_argument("--hidden-key", required=True, type=Path)
    parser.add_argument("--keys-py", required=True, type=Path)
    parser.add_argument("--model", default="openai/gpt-5-mini")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--response-cache", type=Path)
    parser.add_argument("--cache-mode", choices=("record", "replay"), default="record")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--timeout-s", type=int, default=180)
    parser.add_argument("--swap-sides", action="store_true")
    args = parser.parse_args(argv)
    client: Any = OpenAICompatibleCategoricalClient.from_openrouter_keys_file(
        args.keys_py,
        model=args.model,
        timeout_s=args.timeout_s,
        max_tokens=8000,
        reasoning_effort="low",
    )
    if args.response_cache is not None:
        client = PersistentCategoricalResponseCacheClient(
            client, args.response_cache, mode=args.cache_mode
        )
    result = evaluate_blinded_preferences(
        packet=_read_json(args.packet),
        hidden_key=_read_json(args.hidden_key),
        client=client,
        batch_size=args.batch_size,
        swap_sides=args.swap_sides,
    )
    _write_json(args.output, result)
    print(
        json.dumps(
            {
                "pair_count": result["pair_count"],
                "accuracy": result["accuracy"],
                "confusion_matrix": result["confusion_matrix"],
                "prediction_counts": result["prediction_counts"],
                "side_swap_applied": result["side_swap_applied"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
