"""Run blinded categorical model-provisional review of transition audits."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any

from steam_video_new.implicit_world_model.l15_graph_navigator.gpt_oss import (
    OpenAICompatibleCategoricalClient,
)

from .transition_audit import REVIEW_LABELS, validate_transition_audit_packet


REVIEW_SCHEMA = "steam-iwm-transition-model-review/v0.1"


def review_transition_audit(
    packet: dict[str, Any],
    client: Any,
    *,
    batch_size: int = 6,
) -> tuple[dict[str, Any], dict[str, Any]]:
    validate_transition_audit_packet(packet)
    if packet.get("annotation_status") != "unreviewed":
        raise ValueError("model review requires an unreviewed packet")
    if batch_size < 1 or batch_size > 26:
        raise ValueError("model review batch size must be between one and twenty-six")
    decisions: list[dict[str, str]] = []
    records = list(packet["records"])
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        aliases = {f"review_{chr(ord('a') + index)}": row for index, row in enumerate(batch)}
        payload = {
            "records": {
                alias: {
                    "question": row["corrected_belief_before"].get("question"),
                    "executed_real_observation": row["executed_real_observation"],
                    "predicted_belief_delta": row["predicted_belief_delta"],
                    "belief_before": _belief_review_view(
                        row["corrected_belief_before"]
                    ),
                    "belief_after_real_correction": _belief_review_view(
                        row["corrected_belief_after"]
                    ),
                }
                for alias, row in aliases.items()
            },
            "allowed_decisions": sorted(REVIEW_LABELS),
            "decision_meanings": {
                "over_crediting": (
                    "predicted progress, role resolution, or answerability exceeds "
                    "what the visible L1 observation supports"
                ),
                "under_crediting": (
                    "prediction misses a belief change directly supported by the "
                    "visible L1 observation"
                ),
                "consistent": (
                    "prediction is categorically compatible with the visible L1 "
                    "observation and justified real correction"
                ),
                "inconclusive": (
                    "the visible L1 observation is too ambiguous to assess the delta"
                ),
            },
            "required_output": {
                "only_key": "decisions",
                "one_row_per_alias": list(aliases),
                "fields": ["decision", "rationale"],
            },
            "contract": {
                "automatic_failure_slice_hidden": True,
                "hidden_clues_and_answers_unavailable": True,
                "judge_only_direct_visible_l1_support": True,
                "categorical_only": True,
                "no_numeric_reward_score_probability_confidence_or_utility": True,
            },
        }
        result = client.complete_json(
            task=(
                "Independently review every action-conditioned transition. Compare "
                "the imagined belief delta with directly visible L1 evidence and the "
                "post-read correction. Return one categorical decision per alias."
            ),
            payload=payload,
        )
        raw = result.get("decisions") if isinstance(result, dict) else None
        if not isinstance(raw, dict) or set(raw) != set(aliases):
            raise ValueError("transition model review coverage mismatch")
        for alias, source in aliases.items():
            row = raw[alias]
            if not isinstance(row, dict) or set(row) != {"decision", "rationale"}:
                raise ValueError("transition model review row schema is invalid")
            decision = str(row["decision"])
            rationale = str(row["rationale"] or "").strip()
            if decision not in REVIEW_LABELS or not rationale:
                raise ValueError("transition model review judgment is invalid")
            decisions.append(
                {
                    "record_id": str(source["record_id"]),
                    "decision": decision,
                    "rationale": rationale,
                }
            )
    if len(decisions) != len(records):
        raise ValueError("transition model review is incomplete")
    review = {
        "schema_version": REVIEW_SCHEMA,
        "packet_id": str(packet["packet_id"]),
        "packet_sha256": _checksum(packet),
        "labels_source": "model_provisional",
        "annotator": str(getattr(client, "model", "unknown")),
        "automatic_failure_slice_visible_to_reviewer": False,
        "hidden_clues_or_answers_visible_to_reviewer": False,
        "decisions": decisions,
        "formal_gate_eligible": False,
        "training_allowed": False,
        "training_performed": False,
    }
    validate_model_review(review, packet)
    inspection = _inspection(packet, review)
    return review, inspection


def validate_model_review(review: dict[str, Any], packet: dict[str, Any]) -> None:
    required = {
        "schema_version",
        "packet_id",
        "packet_sha256",
        "labels_source",
        "annotator",
        "automatic_failure_slice_visible_to_reviewer",
        "hidden_clues_or_answers_visible_to_reviewer",
        "decisions",
        "formal_gate_eligible",
        "training_allowed",
        "training_performed",
    }
    if set(review) != required or review.get("schema_version") != REVIEW_SCHEMA:
        raise ValueError("transition model review schema is invalid")
    if review.get("packet_id") != packet.get("packet_id"):
        raise ValueError("transition model review packet identity mismatch")
    if review.get("packet_sha256") != _checksum(packet):
        raise ValueError("transition model review packet checksum mismatch")
    if review.get("labels_source") != "model_provisional":
        raise ValueError("transition model review source is invalid")
    if not str(review.get("annotator") or "").strip():
        raise ValueError("transition model review annotator is missing")
    if review.get("automatic_failure_slice_visible_to_reviewer") is not False:
        raise ValueError("automatic failure slice leaked to model reviewer")
    if review.get("hidden_clues_or_answers_visible_to_reviewer") is not False:
        raise ValueError("hidden evaluator supervision leaked to model reviewer")
    if any(
        review.get(key) is not False
        for key in ("formal_gate_eligible", "training_allowed", "training_performed")
    ):
        raise ValueError("model-provisional review cannot pass a formal gate")
    rows = review.get("decisions")
    if not isinstance(rows, list):
        raise ValueError("transition model review decisions must be a list")
    expected = {str(row["record_id"]) for row in packet["records"]}
    actual = {str(row.get("record_id") or "") for row in rows}
    if len(rows) != len(actual) or actual != expected:
        raise ValueError("transition model review decision coverage mismatch")
    for row in rows:
        if set(row) != {"record_id", "decision", "rationale"}:
            raise ValueError("transition model review decision schema is invalid")
        if row["decision"] not in REVIEW_LABELS or not str(
            row["rationale"] or ""
        ).strip():
            raise ValueError("transition model review decision is incomplete")


def _belief_review_view(belief: dict[str, Any]) -> dict[str, Any]:
    return {
        "missing_roles": belief.get("missing_roles") or [],
        "grounded_role_evidence": belief.get("grounded_role_evidence") or [],
        "contradictions": belief.get("contradictions") or [],
        "answerability": belief.get("answerability"),
    }


def _inspection(packet: dict[str, Any], review: dict[str, Any]) -> dict[str, Any]:
    automatic = {
        str(row["record_id"]): str(row["automatic_failure_slice"])
        for row in packet["records"]
    }
    counts: Counter[str] = Counter()
    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    for row in review["decisions"]:
        decision = str(row["decision"])
        counts[decision] += 1
        confusion[automatic[str(row["record_id"])]][decision] += 1
    return {
        "schema_version": "steam-iwm-transition-model-review-inspection/v0.1",
        "packet_id": packet["packet_id"],
        "reviewed_record_count": len(review["decisions"]),
        "decision_counts": dict(sorted(counts.items())),
        "automatic_slice_vs_model_review": {
            key: dict(sorted(value.items()))
            for key, value in sorted(confusion.items())
        },
        "coverage_complete": len(review["decisions"]) == len(packet["records"]),
        "independent_human_review_complete": False,
        "formal_gate_eligible": False,
        "training_allowed": False,
        "training_performed": False,
    }


def _checksum(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", required=True, type=Path)
    parser.add_argument("--keys-py", required=True, type=Path)
    parser.add_argument("--model", default="openai/gpt-5.6")
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--timeout-s", type=int, default=180)
    parser.add_argument("--max-tokens", type=int, default=6000)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--inspection-output", required=True, type=Path)
    args = parser.parse_args(argv)
    client = OpenAICompatibleCategoricalClient.from_openrouter_keys_file(
        args.keys_py,
        model=args.model,
        timeout_s=args.timeout_s,
        max_tokens=args.max_tokens,
        reasoning_effort="low",
    )
    review, inspection = review_transition_audit(
        _read(args.packet), client, batch_size=args.batch_size
    )
    _write(args.output, review)
    _write(args.inspection_output, inspection)
    print(json.dumps(inspection, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
