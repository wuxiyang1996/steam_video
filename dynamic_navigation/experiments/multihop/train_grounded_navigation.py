#!/usr/bin/env python3
"""Train pretrained navigators on grounded MuSiQue next-hop trajectories."""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from grounded_navigation import (
    BertNavigator,
    FrozenQwenEmbedder,
    NavigationExample,
    QwenLoRANavigator,
    format_query,
    load_musique_navigation,
    score_paragraphs,
)


class TrajectoryDataset(Dataset):
    def __init__(
        self,
        examples: list[NavigationExample],
        paragraph_embeddings: torch.Tensor,
        paragraph_mask: torch.Tensor,
    ):
        self.examples = examples
        self.paragraph_embeddings = paragraph_embeddings
        self.paragraph_mask = paragraph_mask
        self.items = [
            (example_index, hop)
            for example_index, example in enumerate(examples)
            for hop in range(len(example.support_order))
        ]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        example_index, hop = self.items[index]
        example = self.examples[example_index]
        visited = torch.zeros(
            self.paragraph_embeddings.size(1), dtype=torch.bool
        )
        for position in example.support_order[:hop]:
            visited[position] = True
        visited |= ~self.paragraph_mask[example_index]
        return {
            "example_index": example_index,
            "hop": hop,
            "question": example.question,
            "evidence": [
                example.paragraphs[position]
                for position in example.support_order[:hop]
            ],
            "paragraph_texts": example.paragraphs,
            "support_order": example.support_order,
            "paragraphs": self.paragraph_embeddings[example_index],
            "visited": visited,
            "target": example.support_order[hop],
        }


def collate(batch):
    return {
        "example_index": torch.tensor(
            [item["example_index"] for item in batch], dtype=torch.long
        ),
        "hop": torch.tensor([item["hop"] for item in batch], dtype=torch.long),
        "question": [item["question"] for item in batch],
        "evidence": [item["evidence"] for item in batch],
        "paragraph_texts": [item["paragraph_texts"] for item in batch],
        "support_order": [item["support_order"] for item in batch],
        "paragraphs": torch.stack([item["paragraphs"] for item in batch]),
        "visited": torch.stack([item["visited"] for item in batch]),
        "target": torch.tensor(
            [item["target"] for item in batch], dtype=torch.long
        ),
    }


def prompts_for_batch(batch, intervention: str | None = None):
    evidence = [list(items) for items in batch["evidence"]]
    questions = list(batch["question"])
    conditioned = [index for index, items in enumerate(evidence) if items]
    if intervention == "question_only":
        evidence = [[] for _ in evidence]
    elif intervention == "evidence_only":
        questions = [
            "Continue to the next supporting fact in this reasoning chain."
            for _ in questions
        ]
    elif intervention == "shuffle_evidence" and len(conditioned) > 1:
        last_items = [evidence[index][-1] for index in conditioned]
        last_items = last_items[-1:] + last_items[:-1]
        for index, replacement in zip(conditioned, last_items):
            evidence[index][-1] = replacement
    elif intervention == "wrong_evidence":
        for index in conditioned:
            target = int(batch["target"][index])
            support = set(batch["support_order"][index])
            distractors = [
                position
                for position in range(len(batch["paragraph_texts"][index]))
                if position not in support and position != target
            ]
            if distractors:
                # Same-question, natural-language distractor: a stricter
                # in-distribution intervention than unrelated batch evidence.
                evidence[index][-1] = batch["paragraph_texts"][index][
                    distractors[0]
                ]
    return [
        format_query(question, history)
        for question, history in zip(questions, evidence)
    ]


def trainable_snapshot(model):
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def restore_snapshot(model, snapshot):
    parameters = dict(model.named_parameters())
    for name, value in snapshot.items():
        parameters[name].data.copy_(value.to(parameters[name].device))


@torch.no_grad()
def evaluate_teacher(
    model,
    loader,
    device,
    intervention: str | None = None,
):
    model.eval()
    correct_by_hop = defaultdict(list)
    predictions_by_example = defaultdict(dict)
    conditioned_correct = []
    for batch in loader:
        prompts = prompts_for_batch(batch, intervention)
        queries = model(prompts, device)
        logits = score_paragraphs(
            queries,
            batch["paragraphs"].to(device),
            batch["visited"].to(device),
        )
        predictions = logits.argmax(-1).cpu()
        correct = predictions == batch["target"]
        for index in range(len(prompts)):
            hop = int(batch["hop"][index])
            example_index = int(batch["example_index"][index])
            value = bool(correct[index])
            correct_by_hop[hop].append(value)
            predictions_by_example[example_index][hop] = value
            if hop > 0:
                conditioned_correct.append(value)
    all_values = [
        value for values in correct_by_hop.values() for value in values
    ]
    path_values = [
        all(hops.values()) for hops in predictions_by_example.values()
    ]
    return {
        "hop_recall_at_1": sum(all_values) / len(all_values),
        "conditioned_hop_recall_at_1": (
            sum(conditioned_correct) / len(conditioned_correct)
        ),
        "path_exact_match": sum(path_values) / len(path_values),
        "by_hop": {
            str(hop + 1): sum(values) / len(values)
            for hop, values in sorted(correct_by_hop.items())
        },
    }


@torch.no_grad()
def evaluate_free_running(
    model,
    examples,
    paragraph_embeddings,
    paragraph_mask,
    device,
    batch_size,
):
    model.eval()
    correct_by_hop = defaultdict(list)
    path_values = []
    for start in range(0, len(examples), batch_size):
        group = examples[start : start + batch_size]
        embeddings = paragraph_embeddings[start : start + len(group)].to(device)
        valid_mask = paragraph_mask[start : start + len(group)].to(device)
        evidence = [[] for _ in group]
        visited = ~valid_mask.clone()
        path_correct = [True for _ in group]
        max_hops = max(len(example.support_order) for example in group)
        for hop in range(max_hops):
            active = [
                index
                for index, example in enumerate(group)
                if hop < len(example.support_order)
            ]
            prompts = [
                format_query(group[index].question, evidence[index])
                for index in active
            ]
            queries = model(prompts, device)
            logits = score_paragraphs(
                queries,
                embeddings[active],
                visited[active],
            )
            predictions = logits.argmax(-1)
            for local_index, group_index in enumerate(active):
                prediction = int(predictions[local_index])
                target = group[group_index].support_order[hop]
                is_correct = prediction == target
                correct_by_hop[hop].append(is_correct)
                path_correct[group_index] &= is_correct
                visited[group_index, prediction] = True
                evidence[group_index].append(
                    group[group_index].paragraphs[prediction]
                )
        path_values.extend(path_correct)
    all_values = [
        value for values in correct_by_hop.values() for value in values
    ]
    return {
        "hop_recall_at_1": sum(all_values) / len(all_values),
        "path_exact_match": sum(path_values) / len(path_values),
        "by_hop": {
            str(hop + 1): sum(values) / len(values)
            for hop, values in sorted(correct_by_hop.items())
        },
    }


@torch.no_grad()
def evaluate_direct(
    queries,
    examples,
    paragraph_embeddings,
    paragraph_mask,
):
    correct_by_hop = defaultdict(list)
    paths = []
    for index, example in enumerate(examples):
        visited = ~paragraph_mask[index].clone()
        path = True
        for hop, target in enumerate(example.support_order):
            logits = score_paragraphs(
                queries[index : index + 1].float(),
                paragraph_embeddings[index : index + 1].float(),
                visited.unsqueeze(0),
            )
            prediction = int(logits.argmax(-1))
            value = prediction == target
            correct_by_hop[hop].append(value)
            path &= value
            visited[prediction] = True
        paths.append(path)
    values = [value for group in correct_by_hop.values() for value in group]
    return {
        "hop_recall_at_1": sum(values) / len(values),
        "path_exact_match": sum(paths) / len(paths),
        "by_hop": {
            str(hop + 1): sum(group) / len(group)
            for hop, group in sorted(correct_by_hop.items())
        },
    }


def train_navigator(
    name,
    model,
    train_loader,
    val_loader,
    device,
    epochs,
    lr,
    grad_accumulation,
):
    model.to(device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.01)
    best_score = -1.0
    best_snapshot = None
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        total_loss = 0.0
        total_count = 0
        for step, batch in enumerate(train_loader):
            prompts = prompts_for_batch(batch)
            with torch.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
            ):
                queries = model(prompts, device)
                logits = score_paragraphs(
                    queries,
                    batch["paragraphs"].to(device),
                    batch["visited"].to(device),
                )
                loss = F.cross_entropy(
                    logits, batch["target"].to(device)
                ) / grad_accumulation
            loss.backward()
            if (step + 1) % grad_accumulation == 0:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                optimizer.zero_grad()
            count = len(prompts)
            total_loss += float(loss.detach()) * grad_accumulation * count
            total_count += count
        if len(train_loader) % grad_accumulation:
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            optimizer.zero_grad()
        validation = evaluate_teacher(model, val_loader, device)
        score = validation["conditioned_hop_recall_at_1"]
        print(
            f"{name} epoch {epoch + 1}/{epochs} "
            f"loss={total_loss / total_count:.4f} "
            f"val_hop={validation['hop_recall_at_1']:.4f} "
            f"val_conditioned={score:.4f}",
            flush=True,
        )
        if score > best_score:
            best_score = score
            best_snapshot = trainable_snapshot(model)
    assert best_snapshot is not None
    restore_snapshot(model, best_snapshot)
    return model


def precompute_embeddings(
    train_examples,
    val_examples,
    device,
    cache_path,
    batch_size,
):
    if cache_path.exists():
        print("loading embedding cache", cache_path, flush=True)
        return torch.load(cache_path, map_location="cpu", weights_only=True)

    embedder = FrozenQwenEmbedder(device)
    all_examples = train_examples + val_examples
    unique_paragraphs = []
    text_to_index = {}
    for example in all_examples:
        for paragraph in example.paragraphs:
            if paragraph not in text_to_index:
                text_to_index[paragraph] = len(unique_paragraphs)
                unique_paragraphs.append(paragraph)
    print(f"encoding {len(unique_paragraphs)} unique paragraphs", flush=True)
    unique_embeddings = embedder.encode(unique_paragraphs, batch_size)
    max_paragraphs = max(len(example.paragraphs) for example in all_examples)
    paragraph_embeddings = torch.zeros(
        len(all_examples),
        max_paragraphs,
        unique_embeddings.size(-1),
        dtype=torch.float16,
    )
    paragraph_mask = torch.zeros(
        len(all_examples), max_paragraphs, dtype=torch.bool
    )
    for index, example in enumerate(all_examples):
        positions = [text_to_index[text] for text in example.paragraphs]
        paragraph_embeddings[index, : len(positions)] = unique_embeddings[positions]
        paragraph_mask[index, : len(positions)] = True
    questions = [format_query(example.question, []) for example in all_examples]
    print(f"encoding {len(questions)} direct queries", flush=True)
    direct_queries = embedder.encode(questions, batch_size)
    cache = {
        "paragraph_embeddings": paragraph_embeddings,
        "paragraph_mask": paragraph_mask,
        "direct_queries": direct_queries,
        "n_train": len(train_examples),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, cache_path)
    del embedder
    torch.cuda.empty_cache()
    return cache


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-train", type=int, default=4096)
    parser.add_argument("--n-val", type=int, default=512)
    parser.add_argument("--min-hops", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--qwen-batch-size", type=int, default=8)
    parser.add_argument("--bert-batch-size", type=int, default=16)
    parser.add_argument("--encode-batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cache", default="outputs/grounded_embeddings_4k.pt")
    parser.add_argument("--out", default="outputs/grounded_navigation_4k.json")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device, flush=True)

    train_examples = load_musique_navigation(
        "train", args.n_train, args.seed, args.min_hops
    )
    val_examples = load_musique_navigation(
        "validation", args.n_val, args.seed + 1, args.min_hops
    )
    cache = precompute_embeddings(
        train_examples,
        val_examples,
        device,
        Path(args.cache),
        args.encode_batch_size,
    )
    split = cache["n_train"]
    train_embeddings = cache["paragraph_embeddings"][:split]
    val_embeddings = cache["paragraph_embeddings"][split:]
    train_mask = cache["paragraph_mask"][:split]
    val_mask = cache["paragraph_mask"][split:]
    val_direct_queries = cache["direct_queries"][split:]
    output_dim = train_embeddings.size(-1)

    train_dataset = TrajectoryDataset(
        train_examples, train_embeddings, train_mask
    )
    val_dataset = TrajectoryDataset(val_examples, val_embeddings, val_mask)
    results = {
        "config": vars(args),
        "data": {
            "train_examples": len(train_examples),
            "train_trajectories": len(train_dataset),
            "val_examples": len(val_examples),
            "val_trajectories": len(val_dataset),
        },
        "direct_qwen": evaluate_direct(
            val_direct_queries, val_examples, val_embeddings, val_mask
        ),
        "models": {},
    }
    print("direct", json.dumps(results["direct_qwen"], indent=2), flush=True)

    specifications = [
        (
            "qwen_lora",
            lambda: QwenLoRANavigator(),
            args.qwen_batch_size,
            1e-4,
            2,
        ),
        (
            "bert_base",
            lambda: BertNavigator(output_dim),
            args.bert_batch_size,
            2e-5,
            1,
        ),
    ]
    for name, factory, batch_size, lr, accumulation in specifications:
        print(f"\n=== {name} ===", flush=True)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate,
            num_workers=0,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate,
            num_workers=0,
        )
        model = train_navigator(
            name,
            factory(),
            train_loader,
            val_loader,
            device,
            args.epochs,
            lr,
            accumulation,
        )
        normal = evaluate_teacher(model, val_loader, device)
        question_only = evaluate_teacher(
            model, val_loader, device, "question_only"
        )
        evidence_only = evaluate_teacher(
            model, val_loader, device, "evidence_only"
        )
        shuffled = evaluate_teacher(
            model, val_loader, device, "shuffle_evidence"
        )
        wrong = evaluate_teacher(
            model, val_loader, device, "wrong_evidence"
        )
        free = evaluate_free_running(
            model,
            val_examples,
            val_embeddings,
            val_mask,
            device,
            batch_size,
        )
        conditioned = normal["conditioned_hop_recall_at_1"]
        result = {
            "teacher_forced": normal,
            "free_running": free,
            "interventions": {
                "question_only": question_only,
                "evidence_only": evidence_only,
                "shuffle_evidence": shuffled,
                "wrong_evidence": wrong,
            },
            "causal_drops": {
                "remove_evidence": conditioned
                - question_only["conditioned_hop_recall_at_1"],
                "remove_question": conditioned
                - evidence_only["conditioned_hop_recall_at_1"],
                "shuffle_evidence": conditioned
                - shuffled["conditioned_hop_recall_at_1"],
                "wrong_evidence": conditioned
                - wrong["conditioned_hop_recall_at_1"],
            },
        }
        results["models"][name] = result
        print(json.dumps(result, indent=2), flush=True)
        del model
        torch.cuda.empty_cache()

    qwen = results["models"]["qwen_lora"]
    bert = results["models"]["bert_base"]
    results["go_nogo"] = {
        "qwen_lora": {
            "beats_direct": qwen["teacher_forced"]["hop_recall_at_1"]
            > results["direct_qwen"]["hop_recall_at_1"],
            "shuffle_drop_ge_5pts": qwen["causal_drops"]["shuffle_evidence"]
            >= 0.05,
            "wrong_drop_ge_5pts": qwen["causal_drops"]["wrong_evidence"] >= 0.05,
        },
        "bert_base": {
            "beats_direct": bert["teacher_forced"]["hop_recall_at_1"]
            > results["direct_qwen"]["hop_recall_at_1"],
            "shuffle_drop_ge_5pts": bert["causal_drops"]["shuffle_evidence"]
            >= 0.05,
            "wrong_drop_ge_5pts": bert["causal_drops"]["wrong_evidence"] >= 0.05,
        },
    }
    for criteria in results["go_nogo"].values():
        criteria["verdict"] = (
            "GO"
            if all(value for key, value in criteria.items() if key != "verdict")
            else "NO-GO"
        )
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2))
    print("wrote", output, flush=True)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
