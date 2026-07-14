"""Grounded, evidence-conditioned navigators for MuSiQue paragraph paths."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModel, AutoTokenizer


QWEN_MODEL = "Qwen/Qwen3-Embedding-0.6B"
BERT_MODEL = "bert-base-uncased"


@dataclass
class NavigationExample:
    id: str
    question: str
    paragraphs: list[str]
    support_order: list[int]
    bridge_answers: list[str]


def load_musique_navigation(
    split: str,
    n: int,
    seed: int,
    min_hops: int = 3,
) -> list[NavigationExample]:
    dataset = load_dataset("dgslibisey/MuSiQue", split=split)
    dataset = dataset.filter(
        lambda row: row.get("answerable", True)
        and len(row.get("question_decomposition") or []) >= min_hops
    ).shuffle(seed=seed)
    if n > 0:
        dataset = dataset.select(range(min(n, len(dataset))))

    examples = []
    for row in dataset:
        paragraphs = sorted(row["paragraphs"], key=lambda item: int(item["idx"]))
        index_to_position = {
            int(paragraph["idx"]): position
            for position, paragraph in enumerate(paragraphs)
        }
        texts = [
            f"{paragraph['title']}. {paragraph['paragraph_text']}"
            for paragraph in paragraphs
        ]
        decomposition = row["question_decomposition"]
        support_order = [
            index_to_position[int(step["paragraph_support_idx"])]
            for step in decomposition
        ]
        bridge_answers = [
            str(step.get("answer") or "").strip() for step in decomposition
        ]
        examples.append(
            NavigationExample(
                id=str(row["id"]),
                question=str(row["question"]).strip(),
                paragraphs=texts,
                support_order=support_order,
                bridge_answers=bridge_answers,
            )
        )
    return examples


def format_query(
    question: str,
    evidence: list[str],
    bridge_answers: list[str] | None = None,
) -> str:
    parts = [
        "Instruct: Retrieve the next supporting paragraph needed to solve "
        "the multi-hop question.",
        f"Question: {question}",
    ]
    if evidence:
        parts.append("Evidence already found:")
        for index, paragraph in enumerate(evidence):
            parts.append(f"[{index + 1}] {paragraph}")
            if bridge_answers and index < len(bridge_answers):
                answer = bridge_answers[index]
                if answer:
                    parts.append(f"Bridge result: {answer}")
    else:
        parts.append("Evidence already found: none")
    return "\n".join(parts)


def pool_last(
    hidden: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    lengths = attention_mask.sum(dim=1).sub(1).clamp_min(0)
    indices = lengths.view(-1, 1, 1).expand(-1, 1, hidden.size(-1))
    return hidden.gather(1, indices).squeeze(1)


def pool_mean(
    hidden: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)


class FrozenQwenEmbedder:
    def __init__(
        self,
        device: str,
        max_length: int = 384,
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.device = torch.device(device)
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(
            QWEN_MODEL, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        self.model = AutoModel.from_pretrained(
            QWEN_MODEL,
            trust_remote_code=True,
            torch_dtype=dtype,
        ).to(self.device)
        self.model.eval()
        self.out_dim = int(self.model.config.hidden_size)

    @torch.no_grad()
    def encode(
        self, texts: list[str], batch_size: int = 32
    ) -> torch.Tensor:
        outputs = []
        for start in range(0, len(texts), batch_size):
            tokens = self.tokenizer(
                texts[start : start + batch_size],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            tokens = {key: value.to(self.device) for key, value in tokens.items()}
            hidden = self.model(**tokens).last_hidden_state
            pooled = pool_last(hidden, tokens["attention_mask"])
            outputs.append(F.normalize(pooled.float(), dim=-1).cpu().half())
        return torch.cat(outputs)


class QwenLoRANavigator(nn.Module):
    def __init__(
        self,
        max_length: int = 512,
        rank: int = 8,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        # This environment has an obsolete AutoAWQ package that PEFT detects
        # even though the base model is not quantized. Disable that irrelevant
        # dispatcher so standard torch Linear layers receive LoRA adapters.
        import peft.tuners.lora.awq as peft_awq

        peft_awq.is_auto_awq_available = lambda: False
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(
            QWEN_MODEL, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        base = AutoModel.from_pretrained(
            QWEN_MODEL,
            trust_remote_code=True,
            torch_dtype=dtype,
        )
        config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=rank,
            lora_alpha=rank * 2,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="none",
        )
        self.model = get_peft_model(base, config)
        self.out_dim = int(base.config.hidden_size)

    def encode(self, texts: list[str], device: torch.device) -> torch.Tensor:
        tokens = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        tokens = {key: value.to(device) for key, value in tokens.items()}
        hidden = self.model(**tokens).last_hidden_state
        return F.normalize(pool_last(hidden, tokens["attention_mask"]).float(), dim=-1)

    def forward(self, texts: list[str], device: torch.device) -> torch.Tensor:
        return self.encode(texts, device)


class BertNavigator(nn.Module):
    def __init__(self, output_dim: int, max_length: int = 512):
        super().__init__()
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)
        self.model = AutoModel.from_pretrained(BERT_MODEL)
        self.projection = nn.Linear(self.model.config.hidden_size, output_dim)

    def encode(self, texts: list[str], device: torch.device) -> torch.Tensor:
        tokens = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        tokens = {key: value.to(device) for key, value in tokens.items()}
        hidden = self.model(**tokens).last_hidden_state
        pooled = pool_mean(hidden, tokens["attention_mask"])
        return F.normalize(self.projection(pooled), dim=-1)

    def forward(self, texts: list[str], device: torch.device) -> torch.Tensor:
        return self.encode(texts, device)


def score_paragraphs(
    queries: torch.Tensor,
    paragraphs: torch.Tensor,
    visited: torch.Tensor | None = None,
    temperature: float = 20.0,
) -> torch.Tensor:
    logits = temperature * torch.einsum(
        "bd,bpd->bp",
        F.normalize(queries.float(), dim=-1),
        F.normalize(paragraphs.float(), dim=-1),
    )
    if visited is not None:
        logits = logits.masked_fill(visited.bool(), -1e4)
    return logits
