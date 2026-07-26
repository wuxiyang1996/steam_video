"""Run a trained transition LoRA on held-out categorical records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from .schemas import TRANSITION_TASK, validate_sft_record
from .serialization import render_sft_example
from .peft_compat import disable_incompatible_unused_torchao_dispatch


FIELDS = (
    "observation_outcome",
    "progress",
    "required_clue_coverage_after",
    "answerability_after",
)


def _load_records(path: Path, split: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            validate_sft_record(row)
            if row["task"] == TRANSITION_TASK and row["split"] == split:
                rows.append(row)
    if not rows:
        raise ValueError(f"no {split} transition records")
    return rows


def _extract_json(text: str) -> Mapping[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        decoder = json.JSONDecoder()
        if start < 0:
            raise
        value, _ = decoder.raw_decode(stripped[start:])
    if not isinstance(value, Mapping):
        raise ValueError("generated transition must be a JSON object")
    return value


def _categorical_prediction(value: Mapping[str, Any]) -> dict[str, str]:
    audit = value.get("categorical_audit")
    if not isinstance(audit, Mapping):
        raise ValueError("generated transition is missing categorical_audit")
    result: dict[str, str] = {}
    for field in FIELDS:
        label = audit.get(field)
        if not isinstance(label, str) or not label:
            raise ValueError(f"generated transition is missing {field}")
        result[field] = label
    return result


def predict(
    records_path: Path,
    *,
    model_name: str,
    adapter_path: Path,
    split: str,
    output_path: Path,
    report_path: Path,
    max_input_length: int,
    max_new_tokens: int,
    batch_size: int = 1,
) -> dict[str, Any]:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows = _load_records(records_path, split)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype="auto",
        device_map="auto",
    )
    torchao_dispatch_disabled = disable_incompatible_unused_torchao_dispatch(model)
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    device = next(model.parameters()).device
    outputs: list[dict[str, Any]] = []
    valid_count = 0
    with torch.inference_mode():
        for batch_start in range(0, len(rows), batch_size):
            batch_rows = rows[batch_start : batch_start + batch_size]
            prompts = [
                tokenizer.apply_chat_template(
                    render_sft_example(row)["messages"][:2],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for row in batch_rows
            ]
            encoded = tokenizer(
                prompts,
                return_tensors="pt",
                add_special_tokens=False,
                padding=True,
                truncation=True,
                max_length=max_input_length,
            ).to(device)
            generated = model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            suffixes = generated[:, encoded["input_ids"].shape[1] :]
            raw_texts = tokenizer.batch_decode(suffixes, skip_special_tokens=True)
            for offset, (row, raw_text) in enumerate(zip(batch_rows, raw_texts)):
                error: str | None = None
                try:
                    categorical = _categorical_prediction(_extract_json(raw_text))
                    valid_count += 1
                except (ValueError, json.JSONDecodeError) as exc:
                    categorical = {field: "invalid_generation" for field in FIELDS}
                    error = f"{type(exc).__name__}: {exc}"
                outputs.append(
                    {
                        "record_id": row["record_id"],
                        "prediction": categorical,
                        "generation_valid": error is None,
                        "generation_error": error,
                        "raw_generation": raw_text,
                        "model_output_contains_numeric_scalar": _contains_numeric_scalar(
                            _extract_json(raw_text) if error is None else {}
                        ),
                    }
                )
                print(
                    json.dumps(
                        {
                            "completed": batch_start + offset + 1,
                            "total": len(rows),
                            "valid": valid_count,
                        }
                    ),
                    flush=True,
                )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        for row in outputs:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    report = {
        "schema_version": "steam-iwm-transition-inference/v0.1",
        "model_name": model_name,
        "adapter_path": str(adapter_path),
        "split": split,
        "record_count": len(rows),
        "valid_generation_count": valid_count,
        "invalid_generation_count": len(rows) - valid_count,
        "numeric_model_output_count": sum(
            row["model_output_contains_numeric_scalar"] for row in outputs
        ),
        "categorical_output_only_required": True,
        "batch_size": batch_size,
        "incompatible_unused_torchao_dispatch_disabled": torchao_dispatch_disabled,
        "training_performed": False,
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def _contains_numeric_scalar(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return True
    if isinstance(value, Mapping):
        return any(_contains_numeric_scalar(child) for child in value.values())
    if isinstance(value, list):
        return any(_contains_numeric_scalar(child) for child in value)
    return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", required=True, type=Path)
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--adapter", required=True, type=Path)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--max-input-length", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args(argv)
    report = predict(
        args.records,
        model_name=args.model,
        adapter_path=args.adapter,
        split=args.split,
        output_path=args.output,
        report_path=args.report,
        max_input_length=args.max_input_length,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["invalid_generation_count"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
