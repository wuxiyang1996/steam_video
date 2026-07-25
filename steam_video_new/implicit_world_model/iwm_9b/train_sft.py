"""LoRA SFT entry point for either IWM transitions or Planner preferences.

Framework imports are intentionally lazy so dataset/readiness checks work in
the lightweight repository environment without installing the training stack.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

from .schemas import PLANNER_TASK, TASKS, validate_sft_record
from .serialization import render_sft_example


def load_training_records(path: Path, task: str) -> list[dict[str, Any]]:
    if task not in TASKS:
        raise ValueError(f"unsupported task: {task}")
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            validate_sft_record(value)
            if value["task"] != task:
                continue
            if value["split"] != "train" or value["training_eligible"] is not True:
                continue
            records.append(value)
    if not records:
        raise ValueError(f"no eligible train records for {task}")
    if task == PLANNER_TASK:
        labels = {str(row["target"]["preference"]["value"]) for row in records}
        if len(labels) < 2:
            raise ValueError(
                "Planner SFT requires at least two train-split preference classes"
            )
    return records


def preflight_report(records: list[dict[str, Any]], task: str) -> dict[str, Any]:
    labels: Counter[str] = Counter()
    if task == PLANNER_TASK:
        labels.update(str(row["target"]["preference"]["value"]) for row in records)
    rendered = [render_sft_example(row) for row in records]
    return {
        "schema_version": "steam-iwm-9b-sft-preflight/v0.1",
        "task": task,
        "train_record_count": len(records),
        "video_count": len({str(row["video_id"]) for row in records}),
        "preference_label_counts": dict(sorted(labels.items())),
        "rendered_example_count": len(rendered),
        "numeric_reward_present": False,
        "training_performed": False,
    }


def train_lora(
    records: list[dict[str, Any]],
    *,
    task: str,
    model_name: str,
    output_dir: Path,
    max_length: int,
    epochs: float,
    learning_rate: float,
    batch_size: int,
    gradient_accumulation_steps: int,
) -> dict[str, Any]:
    try:
        import torch
        from peft import LoraConfig, get_peft_model
        from torch.utils.data import Dataset
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            DataCollatorForSeq2Seq,
            Trainer,
            TrainingArguments,
        )
    except ImportError as error:
        raise RuntimeError(
            "9B SFT dependencies are missing; install requirements-sft.txt "
            "in an isolated training environment"
        ) from error

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    class _ChatDataset(Dataset):
        def __init__(self, rows: list[dict[str, Any]]) -> None:
            self.examples = [
                _tokenize_record(row, tokenizer=tokenizer, max_length=max_length)
                for row in rows
            ]

        def __len__(self) -> int:
            return len(self.examples)

        def __getitem__(self, index: int) -> dict[str, list[int]]:
            return self.examples[index]

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype="auto",
    )
    model.config.use_cache = False
    model = get_peft_model(
        model,
        LoraConfig(
            task_type="CAUSAL_LM",
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        ),
    )
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    bf16 = bool(
        torch.cuda.is_available()
        and getattr(torch.cuda, "is_bf16_supported", lambda: False)()
    )
    arguments = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        bf16=bf16,
        fp16=bool(torch.cuda.is_available() and not bf16),
        gradient_checkpointing=True,
        logging_steps=1,
        save_strategy="epoch",
        report_to=[],
        remove_unused_columns=False,
        seed=17,
        data_seed=17,
    )
    dataset = _ChatDataset(records)
    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100,
        return_tensors="pt",
    )
    trainer = Trainer(
        model=model,
        args=arguments,
        train_dataset=dataset,
        data_collator=collator,
    )
    result = trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    report = {
        **preflight_report(records, task),
        "model_name": model_name,
        "adapter_kind": f"{task}_adapter",
        "max_length": max_length,
        "train_metrics": dict(result.metrics),
        "training_performed": True,
    }
    (output_dir / "training_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def _tokenize_record(
    record: dict[str, Any],
    *,
    tokenizer: Any,
    max_length: int,
) -> dict[str, list[int]]:
    rendered = render_sft_example(record)
    messages = rendered["messages"]
    prompt = tokenizer.apply_chat_template(
        messages[:2],
        tokenize=False,
        add_generation_prompt=True,
    )
    target = str(messages[2]["content"]) + str(tokenizer.eos_token or "")
    prompt_ids = tokenizer(
        prompt,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
    )["input_ids"]
    full_ids = tokenizer(
        prompt + target,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
    )["input_ids"]
    if len(full_ids) <= len(prompt_ids):
        raise ValueError(
            f"max_length={max_length} truncates the entire target for "
            f"{record['record_id']}"
        )
    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids) :]
    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--task", choices=sorted(TASKS), required=True)
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    records = load_training_records(args.records, args.task)
    if args.dry_run:
        report = preflight_report(records, args.task)
    else:
        report = train_lora(
            records,
            task=args.task,
            model_name=args.model,
            output_dir=args.output_dir,
            max_length=args.max_length,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
        )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
