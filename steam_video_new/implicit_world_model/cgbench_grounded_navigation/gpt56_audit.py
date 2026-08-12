"""Run blinded categorical GPT-5.6 visual audit over CG-Bench contact sheets."""

from __future__ import annotations

import argparse
import base64
import json
import runpy
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from .builder import _write_json


AUDIT_MODEL = "openai/gpt-5.6-sol"
TRIAGE = {"accept", "reject", "inconclusive"}
RELEVANCE = {"relevant", "unrelated", "inconclusive"}
EVIDENCE_KIND = {"visual", "readable_text", "both", "none"}


def run_gpt56_audit(
    packet: dict[str, Any], *, packet_root: Path, api_key: str,
    model: str = AUDIT_MODEL, workers: int = 4,
    checkpoint_path: Path | None = None,
) -> dict[str, Any]:
    """Review public evidence only; clue/control labels are intentionally absent."""
    items = packet.get("items") or []
    pending = [item.get("review_id") for item in items
               if (item.get("observation") or {}).get("descriptor_status")
               != "grounded_qwen_vl_read"]
    if pending:
        raise ValueError(f"GPT-5.6 audit requires grounded Qwen observations; pending={len(pending)}")
    decisions: dict[str, dict[str, Any]] = {}
    if checkpoint_path is not None and checkpoint_path.is_file():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint.get("model") != model or checkpoint.get("dataset_id") != packet.get("dataset_id"):
            raise ValueError("GPT-5.6 checkpoint provenance does not match the packet")
        decisions = {
            str(row["review_id"]): row for row in checkpoint.get("decisions") or []
            if row.get("request_status") == "completed"
        }
    lock = threading.Lock()

    def persist() -> None:
        if checkpoint_path is None:
            return
        payload = _audit_payload(packet, model, decisions, status="running_checkpoint")
        _write_json(checkpoint_path, payload)

    def review(item: dict[str, Any]) -> dict[str, Any]:
        ref = str(item.get("contact_sheet_ref") or "")
        image_path = (packet_root / ref).resolve()
        if packet_root.resolve() not in image_path.parents or not image_path.is_file():
            raise ValueError(f"contact sheet unavailable for {item.get('review_id')}")
        first_error: Exception | None = None
        for _ in range(2):
            try:
                result = _request_review(item, image_path=image_path, api_key=api_key, model=model)
                return _validate_decision(str(item["review_id"]), result)
            except Exception as exc:
                if first_error is None:
                    first_error = exc
        assert first_error is not None
        raise first_error

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(review, item): item for item in items
                   if str(item["review_id"]) not in decisions}
        for future in as_completed(futures):
            item = futures[future]
            try:
                decision = future.result()
            except Exception as exc:  # retain a categorical failure for audit coverage
                decision = {
                    "review_id": item["review_id"], "media_alignment": "inconclusive",
                    "descriptor_grounding": "inconclusive", "question_relevance": "inconclusive",
                    "evidence_kind": "none", "rationale": f"API or schema failure: {type(exc).__name__}",
                    "request_status": "failed",
                }
            with lock:
                decisions[str(item["review_id"])] = decision
                persist()
    if len(decisions) != len(items):
        raise ValueError("GPT-5.6 audit did not cover every packet item")
    return _audit_payload(packet, model, decisions, status="model_provisional")


def inspect_audit_against_hidden_key(
    audit: dict[str, Any], hidden: dict[str, Any]
) -> dict[str, Any]:
    """Reveal labels only after decisions are frozen; report counts, never train labels."""
    roles = {row["review_id"]: row["gt_interval_role"] for row in hidden.get("items") or []}
    table: dict[str, dict[str, int]] = {
        role: {label: 0 for label in sorted(RELEVANCE)} for role in ("clue", "control")
    }
    media_counts = {label: 0 for label in sorted(TRIAGE)}
    descriptor_counts = {label: 0 for label in sorted(TRIAGE)}
    failed = 0
    for decision in audit.get("decisions") or []:
        review_id = str(decision.get("review_id") or "")
        role = roles.get(review_id)
        if role not in table:
            raise ValueError(f"hidden key is missing {review_id}")
        label = str(decision.get("question_relevance") or "")
        if label not in RELEVANCE:
            raise ValueError(f"invalid frozen relevance for {review_id}")
        table[role][label] += 1
        media_counts[str(decision["media_alignment"])] += 1
        descriptor_counts[str(decision["descriptor_grounding"])] += 1
        failed += decision.get("request_status") == "failed"
    return {
        "schema_version": "steam-cgbench-gpt56-audit-inspection/v0.1",
        "model": audit.get("model"), "labels_source": "model_provisional",
        "role_by_relevance_counts": table, "request_failure_count": failed,
        "media_alignment_counts": media_counts,
        "descriptor_grounding_counts": descriptor_counts,
        "control_contamination_definition": "control judged relevant by blinded reviewer",
        "clue_miss_definition": "clue judged unrelated by blinded reviewer",
        "separate_gate_status": {
            "model_review_coverage": (
                "passed" if len(audit.get("decisions") or []) == len(roles) and not failed else "failed"
            ),
            "media_alignment": "report_categorical_distribution_separately",
            "descriptor_grounding": "report_categorical_distribution_separately",
            "control_purity": "requires_human_calibrated_threshold",
            "clue_recall": "requires_human_calibrated_threshold",
            "human_lock": "pending",
        },
        "formal_eligible": False,
        "formal_blocker": "independent human review is required",
    }


def _request_review(item: dict[str, Any], *, image_path: Path, api_key: str,
                    model: str) -> dict[str, Any]:
    import requests
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    prompt = {
        "question": item.get("question"), "choices": item.get("choices"),
        "interval": item.get("interval"), "qwen_observation": item.get("observation"),
        "criteria": {
            "media_alignment": "Does the contact sheet depict the declared interval without corruption?",
            "descriptor_grounding": "Are Qwen's factual claims directly supported by visible frames/text?",
            "question_relevance": (
                "relevant if the interval contributes any partial fact needed for the answer; "
                "it need not answer the whole question alone"
            ),
            "evidence_kind": "visual|readable_text|both|none",
        },
        "required_output": {
            "media_alignment": "accept|reject|inconclusive",
            "descriptor_grounding": "accept|reject|inconclusive",
            "question_relevance": "relevant|unrelated|inconclusive",
            "evidence_kind": "visual|readable_text|both|none",
            "rationale": "short evidence-grounded sentence",
        },
    }
    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": model, "temperature": 0, "max_tokens": 400,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": (
                    "You are a blinded video-evidence auditor. You never see or infer hidden "
                    "clue/control labels. Return one JSON object using only categorical fields; "
                    "do not output confidence, probability, score, utility, or any number."
                )},
                {"role": "user", "content": [
                    {"type": "text", "text": json.dumps(prompt, ensure_ascii=False)},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}},
                ]},
            ],
        }, timeout=180,
    )
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]
    return json.loads(content)


def _validate_decision(review_id: str, result: dict[str, Any]) -> dict[str, Any]:
    if _contains_number(result):
        raise ValueError("GPT-5.6 audit output contains a forbidden numeric value")
    if result.get("media_alignment") not in TRIAGE:
        raise ValueError("invalid media_alignment")
    if result.get("descriptor_grounding") not in TRIAGE:
        raise ValueError("invalid descriptor_grounding")
    if result.get("question_relevance") not in RELEVANCE:
        raise ValueError("invalid question_relevance")
    if result.get("evidence_kind") not in EVIDENCE_KIND:
        raise ValueError("invalid evidence_kind")
    rationale = str(result.get("rationale") or "").strip()
    if not rationale:
        raise ValueError("missing rationale")
    return {"review_id": review_id, "media_alignment": result["media_alignment"],
            "descriptor_grounding": result["descriptor_grounding"],
            "question_relevance": result["question_relevance"],
            "evidence_kind": result["evidence_kind"], "rationale": rationale,
            "request_status": "completed"}


def _contains_number(value: Any) -> bool:
    if isinstance(value, bool) or value is None:
        return False
    if isinstance(value, (int, float)):
        return True
    if isinstance(value, dict):
        return any(_contains_number(child) for child in value.values())
    if isinstance(value, list):
        return any(_contains_number(child) for child in value)
    return False


def _audit_payload(packet: dict[str, Any], model: str,
                   decisions: dict[str, dict[str, Any]], *, status: str) -> dict[str, Any]:
    return {
        "schema_version": "steam-cgbench-gpt56-visual-audit/v0.1",
        "dataset_id": packet.get("dataset_id"), "labels_source": "model_provisional",
        "model": model, "annotation_status": status,
        "blinding": "reviewer did not receive clue/control identities or terminal answers",
        "decisions": [decisions[key] for key in sorted(decisions)],
        "formal_eligible": False, "training_ready": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", required=True, type=Path)
    parser.add_argument("--hidden-key", required=True, type=Path)
    parser.add_argument("--keys", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--inspection", required=True, type=Path)
    parser.add_argument("--model", default=AUDIT_MODEL)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--asset-root", type=Path)
    args = parser.parse_args(argv)
    namespace = runpy.run_path(str(args.keys.expanduser().resolve()))
    api_key = namespace.get("OPENROUTER_API_KEY")
    if not isinstance(api_key, str) or not api_key.strip():
        raise ValueError("keys file does not define OPENROUTER_API_KEY")
    packet = json.loads(args.packet.read_text(encoding="utf-8"))
    audit = run_gpt56_audit(packet, packet_root=args.asset_root or args.packet.parent, api_key=api_key,
                            model=args.model, workers=args.workers,
                            checkpoint_path=args.output)
    _write_json(args.output, audit)
    hidden = json.loads(args.hidden_key.read_text(encoding="utf-8"))
    inspection = inspect_audit_against_hidden_key(audit, hidden)
    _write_json(args.inspection, inspection)
    print(json.dumps(inspection, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
