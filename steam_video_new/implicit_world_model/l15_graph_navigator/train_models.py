"""Train categorical transition heads and an ordinal-only preference baseline."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any


def train_baselines(
    records: list[dict[str, Any]],
    *,
    output_dir: Path,
    allow_ai_provisional: bool = False,
) -> dict[str, Any]:
    _validate_training_records(records, allow_ai_provisional=allow_ai_provisional)
    transition_rows = [
        row for row in records if row.get("task") == "observation_belief_transition"
    ]
    preference_rows = [
        row for row in records if row.get("task") == "trajectory_pairwise_preference"
    ]
    if not transition_rows or not preference_rows:
        raise ValueError("training requires both transition and preference records")
    preference_labels = {str((row.get("target") or {}).get("label")) for row in preference_rows}
    if len(preference_labels) < 2:
        raise ValueError("preference training requires at least two observed label classes")

    destination = output_dir.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    transition_bundle, transition_report = _train_transition_bundle(transition_rows)
    preference_bundle, preference_report = _train_preference_bundle(preference_rows)

    import joblib

    transition_path = destination / "transition_model.joblib"
    preference_path = destination / "preference_model.joblib"
    joblib.dump(transition_bundle, transition_path)
    joblib.dump(preference_bundle, preference_path)
    report = {
        "schema_version": "steam-preference-model-training/v0.1",
        "output_contract": "categorical_transition_and_ordinal_preference_only",
        "reward_model": False,
        "transition": transition_report,
        "preference": preference_report,
        "artifacts": {
            "transition_model": str(transition_path),
            "preference_model": str(preference_path),
        },
        "trust": {
            "allow_ai_provisional": allow_ai_provisional,
            "label_sources": sorted({str(row.get("label_source")) for row in records}),
        },
    }
    (destination / "training_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def predict_transition_labels(bundle: dict[str, Any], row: dict[str, Any]) -> dict[str, str]:
    """Return categorical labels only; probabilities are intentionally not exposed."""

    text = _transition_text(row)
    matrix = bundle["vectorizer"].transform([text])
    return {
        name: str(model.predict(matrix)[0])
        for name, model in bundle["heads"].items()
    }


def predict_preference_label(bundle: dict[str, Any], row: dict[str, Any]) -> str:
    """Return one four-way preference label, never a scalar action value."""

    matrix = bundle["vectorizer"].transform([_preference_text(row)])
    return str(bundle["classifier"].predict(matrix)[0])


def _train_transition_bundle(
    rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    from sklearn.feature_extraction.text import TfidfVectorizer

    texts = [_transition_text(row) for row in rows]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=20000)
    matrix = vectorizer.fit_transform(texts)
    targets = {
        "observation_role": [
            str(((row.get("target") or {}).get("observation_descriptor") or {}).get("role") or "none")
            for row in rows
        ],
        "resolved_roles": [
            "|".join(
                sorted(
                    str(value)
                    for value in ((row.get("target") or {}).get("belief_delta") or {}).get("resolved_roles") or []
                )
            )
            or "none"
            for row in rows
        ],
        "uncertainty_change": [
            str(((row.get("target") or {}).get("belief_delta") or {}).get("uncertainty_change") or "unchanged")
            for row in rows
        ],
        "answerability_after": [
            str(((row.get("target") or {}).get("belief_delta") or {}).get("answerability_after") or "not_ready")
            for row in rows
        ],
        "contradiction_change": [
            "present"
            if ((row.get("target") or {}).get("belief_delta") or {}).get("contradiction_updates")
            else "absent"
            for row in rows
        ],
    }
    heads = {name: _fit_classifier(matrix, values) for name, values in targets.items()}
    return (
        {"vectorizer": vectorizer, "heads": heads, "output_contract": "categorical_only"},
        {
            "record_count": len(rows),
            "video_count": len({str(row.get("video_id")) for row in rows}),
            "heads": {
                name: {"classes": sorted(set(values)), "class_counts": dict(Counter(values))}
                for name, values in targets.items()
            },
        },
    )


def _train_preference_bundle(
    rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    from sklearn.feature_extraction.text import TfidfVectorizer

    texts = [_preference_text(row) for row in rows]
    labels = [str((row.get("target") or {}).get("label")) for row in rows]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=30000)
    matrix = vectorizer.fit_transform(texts)
    classifier = _fit_classifier(matrix, labels)
    videos = sorted({str(row.get("video_id")) for row in rows})
    held_out = _held_out_videos(videos)
    evaluation = _group_evaluation(texts, labels, [str(row.get("video_id")) for row in rows], held_out)
    return (
        {
            "vectorizer": vectorizer,
            "classifier": classifier,
            "allowed_outputs": ["prefer_left", "tie", "prefer_right", "incomparable"],
            "output_contract": "ordinal_label_only",
        },
        {
            "record_count": len(rows),
            "video_count": len(videos),
            "classes": sorted(set(labels)),
            "class_counts": dict(Counter(labels)),
            "held_out_videos": held_out,
            "video_disjoint_evaluation": evaluation,
        },
    )


def _fit_classifier(matrix: Any, labels: list[str]) -> Any:
    if len(set(labels)) == 1:
        from sklearn.dummy import DummyClassifier

        model = DummyClassifier(strategy="most_frequent")
    else:
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression(max_iter=500, class_weight="balanced")
    return model.fit(matrix, labels)


def _group_evaluation(
    texts: list[str],
    labels: list[str],
    videos: list[str],
    held_out: list[str],
) -> dict[str, Any]:
    if not held_out:
        return {"status": "not_available", "reason": "fewer than three videos"}
    train_indices = [index for index, video in enumerate(videos) if video not in set(held_out)]
    test_indices = [index for index, video in enumerate(videos) if video in set(held_out)]
    train_labels = [labels[index] for index in train_indices]
    if len(set(train_labels)) < 2 or not test_indices:
        return {"status": "not_available", "reason": "video split lacks train label diversity"}
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=30000)
    train_matrix = vectorizer.fit_transform([texts[index] for index in train_indices])
    test_matrix = vectorizer.transform([texts[index] for index in test_indices])
    classifier = _fit_classifier(train_matrix, train_labels)
    predicted = [str(value) for value in classifier.predict(test_matrix)]
    gold = [labels[index] for index in test_indices]
    return {
        "status": "evaluated",
        "train_count": len(train_indices),
        "test_count": len(test_indices),
        "exact_accuracy": sum(left == right for left, right in zip(predicted, gold)) / len(gold),
    }


def _held_out_videos(videos: list[str]) -> list[str]:
    if len(videos) < 3:
        return []
    ordered = sorted(videos, key=lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest())
    return ordered[: max(1, round(len(ordered) * 0.2))]


def _transition_text(row: dict[str, Any]) -> str:
    return _json_text(
        {
            "question": row.get("question"),
            "checkpoint": row.get("checkpoint"),
            "action": row.get("action"),
        }
    )


def _preference_text(row: dict[str, Any]) -> str:
    return _json_text(
        {
            "question": row.get("question"),
            "checkpoint": row.get("checkpoint"),
            "left": row.get("left"),
            "right": row.get("right"),
        }
    )


def _json_text(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _validate_training_records(
    records: list[dict[str, Any]],
    *,
    allow_ai_provisional: bool,
) -> None:
    if not records:
        raise ValueError("training JSONL is empty")
    allowed_labels = {"prefer_left", "tie", "prefer_right", "incomparable"}
    for index, row in enumerate(records):
        source = str(row.get("label_source") or "")
        if source != "human_locked" and not (
            allow_ai_provisional and source == "ai_provisional"
        ):
            raise ValueError(f"record {index} is not human_locked")
        if not row.get("video_id"):
            raise ValueError(f"record {index} lacks video_id for leakage-safe splitting")
        if row.get("task") == "trajectory_pairwise_preference":
            label = str((row.get("target") or {}).get("label") or "")
            if label not in allowed_labels:
                raise ValueError(f"record {index} has invalid preference label: {label}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-jsonl", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--allow-ai-provisional", action="store_true")
    args = parser.parse_args(argv)
    records = [
        json.loads(line)
        for line in args.training_jsonl.expanduser().resolve().read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    report = train_baselines(
        records,
        output_dir=args.output_dir,
        allow_ai_provisional=args.allow_ai_provisional,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
