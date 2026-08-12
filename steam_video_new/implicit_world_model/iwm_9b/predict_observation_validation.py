"""Run an observation-transition LoRA on representation-complete held-out rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from .peft_compat import disable_incompatible_unused_torchao_dispatch
from .predict_validation import _contains_numeric_scalar, _extract_json
from .schemas import OBSERVATION_TASK, validate_sft_record
from .serialization import render_sft_example


def _load_records(path: Path, split: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            validate_sft_record(row)
            if (
                row["task"] == OBSERVATION_TASK
                and row["split"] == split
                and row.get("representation_eligible") is True
            ):
                rows.append(row)
    if not rows:
        raise ValueError(f"no representation-complete {split} observation records")
    return rows


def _observation_prediction(value: Mapping[str, Any]) -> dict[str, Any]:
    descriptor = value.get("observation_descriptor")
    if not isinstance(descriptor, Mapping):
        raise ValueError("generated observation is missing observation_descriptor")
    event = descriptor.get("event")
    entities = descriptor.get("entities")
    states = descriptor.get("states")
    if not isinstance(event, str):
        raise ValueError("generated observation event must be a string")
    if not isinstance(entities, list) or not all(
        isinstance(row, Mapping) for row in entities
    ):
        raise ValueError("generated observation entities must be a list of objects")
    if not isinstance(states, list):
        raise ValueError("generated observation states must be a list")
    return {
        "event": event,
        "entities": [dict(row) for row in entities],
        "states": states,
        "state_change": descriptor.get("state_change"),
    }


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
    numeric_count = 0
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
            texts = tokenizer.batch_decode(suffixes, skip_special_tokens=True)
            for offset, (row, raw_text) in enumerate(zip(batch_rows, texts)):
                error: str | None = None
                contains_numeric = False
                try:
                    parsed = _extract_json(raw_text)
                    descriptor = _observation_prediction(parsed)
                    contains_numeric = _contains_numeric_scalar(parsed)
                    valid_count += 1
                    numeric_count += int(contains_numeric)
                except (ValueError, json.JSONDecodeError) as exc:
                    descriptor = None
                    error = f"{type(exc).__name__}: {exc}"
                outputs.append(
                    {
                        "record_id": row["record_id"],
                        "prediction": {"observation_descriptor": descriptor},
                        "generation_valid": error is None,
                        "generation_error": error,
                        "raw_generation": raw_text,
                        "model_output_contains_numeric_scalar": contains_numeric,
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
        "schema_version": "steam-iwm-observation-inference/v0.1",
        "model_name": model_name,
        "adapter_path": str(adapter_path),
        "split": split,
        "record_count": len(rows),
        "valid_generation_count": valid_count,
        "invalid_generation_count": len(rows) - valid_count,
        "numeric_model_output_count": numeric_count,
        "batch_size": batch_size,
        "representation_complete_only": True,
        "incompatible_unused_torchao_dispatch_disabled": (
            torchao_dispatch_disabled
        ),
        "training_performed": False,
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", required=True, type=Path)
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--adapter", required=True, type=Path)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--max-input-length", type=int, default=1536)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
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
    return 0 if (
        report["invalid_generation_count"] == 0
        and report["numeric_model_output_count"] == 0
    ) else 2


if __name__ == "__main__":
    raise SystemExit(main())
