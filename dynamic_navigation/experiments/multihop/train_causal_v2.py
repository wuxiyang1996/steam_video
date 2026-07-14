#!/usr/bin/env python3
"""Rerun with token-preserving memory and causal navigation v2."""

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

from causal_v2 import (
    CausalNavigatorV2,
    PooledTokenProbe,
    TokenOneShot,
    causal_loss,
    direction_advantage_loss,
    novelty_loss,
    progress_loss,
)
from data import MHExample, load_hotpot_subset, load_musique_subset
from encoder import build_encoder


class TokenDataset(Dataset):
    def __init__(
        self,
        examples: list[MHExample],
        token_states: torch.Tensor,
        token_mask: torch.Tensor,
        question_states: torch.Tensor,
        candidate_states: torch.Tensor,
        targets: torch.Tensor,
    ):
        self.examples = examples
        self.token_states = token_states
        self.token_mask = token_mask
        self.question_states = question_states
        self.candidate_states = candidate_states
        self.targets = targets

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return {
            "tokens": self.token_states[index],
            "mask": self.token_mask[index],
            "question": self.question_states[index],
            "candidates": self.candidate_states[index],
            "target": self.targets[index],
            "qtype": self.examples[index].qtype,
        }


def collate(batch):
    return {
        "tokens": torch.stack([item["tokens"] for item in batch]),
        "mask": torch.stack([item["mask"] for item in batch]),
        "question": torch.stack([item["question"] for item in batch]),
        "candidates": torch.stack([item["candidates"] for item in batch]),
        "target": torch.stack([item["target"] for item in batch]),
        "qtype": [item["qtype"] for item in batch],
    }


def shuffle_candidates(examples: list[MHExample], seed: int):
    rng = random.Random(seed)
    shuffled_texts = []
    targets = []
    for example in examples:
        indexed = list(enumerate(example.candidates))
        rng.shuffle(indexed)
        targets.append(next(i for i, (old_index, _) in enumerate(indexed) if old_index == 0))
        shuffled_texts.append([text for _, text in indexed])
    return shuffled_texts, torch.tensor(targets, dtype=torch.long)


@torch.no_grad()
def precompute(encoder, examples, seed, max_length, encode_batch_size=2):
    contexts = [example.context for example in examples]
    questions = [example.question for example in examples]
    candidate_texts, targets = shuffle_candidates(examples, seed)
    print(f"  token encoding {len(contexts)} contexts...", flush=True)
    tokens, masks = encoder.encode_tokens(
        contexts, batch_size=encode_batch_size, max_length=max_length
    )
    print("  question embeddings...", flush=True)
    question_states = encoder.encode(questions, batch_size=16).cpu()
    flat_candidates = [text for group in candidate_texts for text in group]
    print(f"  candidate embeddings ({len(flat_candidates)})...", flush=True)
    candidate_states = encoder.encode(flat_candidates, batch_size=32).cpu()
    candidate_states = candidate_states.view(
        len(examples), len(candidate_texts[0]), -1
    )
    return (
        tokens,
        masks,
        F.normalize(question_states, dim=-1),
        F.normalize(candidate_states, dim=-1),
        targets,
    )


def move(batch, device):
    return {
        "tokens": batch["tokens"].to(device),
        "mask": batch["mask"].to(device),
        "question": batch["question"].to(device),
        "candidates": batch["candidates"].to(device),
        "targets": batch["target"].to(device),
    }


def train(model, loader, device, epochs, lr, is_causal):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        sums = defaultdict(float)
        count = 0
        for batch in loader:
            inputs = move(batch, device)
            output = model(**inputs)
            qa = F.cross_entropy(output["logits"], inputs["targets"])
            progress = progress_loss(output.get("margins"))
            necessity = (
                causal_loss(output["causal_gains"])
                if is_causal
                else qa.new_zeros(())
            )
            novelty = (
                novelty_loss(output["directions"])
                if is_causal
                else qa.new_zeros(())
            )
            direction_advantage = (
                direction_advantage_loss(output["direction_advantages"])
                if is_causal
                else qa.new_zeros(())
            )
            diversity = output.get("memory_diversity", qa.new_zeros(()))
            loss = qa + 0.4 * progress
            if is_causal:
                loss = (
                    loss
                    + 0.3 * necessity
                    + 0.2 * direction_advantage
                    + 0.03 * novelty
                    + 0.05 * diversity
                )
            elif "memory_diversity" in output:
                loss = loss + 0.05 * diversity
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            size = inputs["targets"].size(0)
            for key, value in {
                "loss": loss,
                "qa": qa,
                "progress": progress,
                "causal": necessity,
                "direction": direction_advantage,
            }.items():
                sums[key] += float(value.detach()) * size
            count += size
        summary = " ".join(f"{key}={value/count:.4f}" for key, value in sums.items())
        print(f"  epoch {epoch+1}/{epochs} {summary}", flush=True)


@torch.no_grad()
def evaluate(model, loader, device, name, intervention=None):
    model.eval()
    predictions, targets, margins, qtypes = [], [], [], []
    gains, gates, advantages = [], [], []
    for batch in loader:
        inputs = move(batch, device)
        if isinstance(model, CausalNavigatorV2):
            inputs["intervention"] = intervention
        output = model(**inputs)
        predictions.append(output["logits"].argmax(-1).cpu())
        targets.append(inputs["targets"].cpu())
        if output.get("margins") is not None:
            margins.append(output["margins"].cpu())
        if output.get("causal_gains") is not None:
            gains.append(output["causal_gains"].cpu())
            gates.append(output["gates"].cpu())
            advantages.append(output["direction_advantages"].cpu())
        qtypes.extend(batch["qtype"])
    prediction = torch.cat(predictions)
    target = torch.cat(targets)
    correct = prediction == target
    result = {
        "model": name,
        "acc": float(correct.float().mean()),
        "n": int(target.numel()),
    }
    if margins:
        margin = torch.cat(margins)
        means = margin.mean(0)
        result["margin_curve"] = [float(value) for value in means]
        result["m_gain"] = float(means[-1] - means[0]) if means.numel() > 1 else 0.0
        result["mono_rate"] = (
            float((margin[:, 1:] > margin[:, :-1]).float().mean())
            if margin.size(1) > 1
            else 0.0
        )
    by_type = defaultdict(list)
    for value, qtype in zip(correct.tolist(), qtypes):
        by_type[qtype].append(float(value))
    result["by_type"] = {
        key: sum(values) / len(values) for key, values in by_type.items()
    }
    if gains:
        gain = torch.cat(gains)
        gate = torch.cat(gates)
        result["causal_gain_mean"] = float(gain.mean())
        result["causal_positive_rate"] = float((gain > 0).float().mean())
        result["gate_mean"] = float(gate.mean())
        advantage = torch.cat(advantages)
        result["direction_advantage_mean"] = float(advantage.mean())
        result["direction_positive_rate"] = float(
            (advantage > 0).float().mean()
        )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", choices=["hotpot", "musique"], default="hotpot"
    )
    parser.add_argument("--encoder", default="qwen3-emb-0.6b")
    parser.add_argument("--n-train", type=int, default=1024)
    parser.add_argument("--n-val", type=int, default=256)
    parser.add_argument("--n-distractors", type=int, default=7)
    parser.add_argument("--min-hops", type=int, default=3)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--encode-batch-size", type=int, default=2)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-latents", type=int, default=32)
    parser.add_argument("--k-steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="outputs/causal_v2_hotpot.json")
    parser.add_argument("--hf-home", default="/fs/gamma-projects/vlm-robot/hf_cache")
    args = parser.parse_args()

    os.environ.setdefault("HF_HOME", args.hf_home)
    os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(args.hf_home, "datasets"))
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} encoder={args.encoder}", flush=True)
    if device == "cuda":
        print("gpu=", torch.cuda.get_device_name(0), flush=True)

    if args.dataset == "musique":
        train_examples = load_musique_subset(
            "train",
            args.n_train,
            args.seed,
            args.n_distractors,
            args.min_hops,
        )
        val_examples = load_musique_subset(
            "validation",
            args.n_val,
            args.seed + 1,
            args.n_distractors,
            args.min_hops,
        )
    else:
        train_examples = load_hotpot_subset(
            "train", args.n_train, args.seed, args.n_distractors
        )
        val_examples = load_hotpot_subset(
            "validation", args.n_val, args.seed + 1, args.n_distractors
        )
    encoder = build_encoder(args.encoder, device)
    print("precompute train...", flush=True)
    train_cache = precompute(
        encoder,
        train_examples,
        args.seed,
        args.max_length,
        args.encode_batch_size,
    )
    print("precompute validation...", flush=True)
    val_cache = precompute(
        encoder,
        val_examples,
        args.seed + 1,
        args.max_length,
        args.encode_batch_size,
    )
    input_dim = encoder.out_dim
    del encoder
    torch.cuda.empty_cache()

    train_loader = DataLoader(
        TokenDataset(train_examples, *train_cache),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
    )
    val_loader = DataLoader(
        TokenDataset(val_examples, *val_cache),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate,
    )
    results = {"config": vars(args), "models": [], "interventions": []}
    causal_model = None
    specifications = [
        ("pooled_token_probe", lambda: PooledTokenProbe(input_dim, args.d_model), False),
        (
            "token_one_shot",
            lambda: TokenOneShot(input_dim, args.d_model, args.n_latents),
            False,
        ),
        (
            "causal_navigation_v2",
            lambda: CausalNavigatorV2(
                input_dim,
                args.d_model,
                args.n_latents,
                args.k_steps,
            ),
            True,
        ),
    ]
    for name, factory, is_causal in specifications:
        print(f"\n=== {name} ===", flush=True)
        model = factory().to(device)
        train(model, train_loader, device, args.epochs, args.lr, is_causal)
        result = evaluate(model, val_loader, device, name)
        print(json.dumps(result, indent=2), flush=True)
        results["models"].append(result)
        if is_causal:
            causal_model = model
        else:
            del model
            torch.cuda.empty_cache()

    assert causal_model is not None
    for intervention in [
        "zero_read",
        "random_direction",
        "shuffle_direction",
        "causal_gate",
        "reject_all",
        "accept_all",
    ]:
        result = evaluate(
            causal_model,
            val_loader,
            device,
            f"causal_navigation_v2/{intervention}",
            intervention,
        )
        print(json.dumps(result, indent=2), flush=True)
        results["interventions"].append(result)

    by_name = {item["model"]: item for item in results["models"]}
    causal = by_name["causal_navigation_v2"]
    one_shot = by_name["token_one_shot"]
    zero = next(item for item in results["interventions"] if item["model"].endswith("zero_read"))
    random_result = next(
        item for item in results["interventions"] if item["model"].endswith("random_direction")
    )
    shuffled = next(
        item
        for item in results["interventions"]
        if item["model"].endswith("shuffle_direction")
    )
    criteria = {
        "causal_gt_token_one_shot": causal["acc"] > one_shot["acc"],
        "zero_read_drop_ge_2pts": causal["acc"] - zero["acc"] >= 0.02,
        "shuffle_direction_drop_ge_2pts": causal["acc"] - shuffled["acc"] >= 0.02,
        "random_direction_reduces_margin": (
            causal["margin_curve"][-1] > random_result["margin_curve"][-1]
        ),
        "margin_rises": causal.get("m_gain", 0) > 0
        and causal.get("mono_rate", 0) > 0.5,
    }
    results["go_nogo"] = {
        "criteria": criteria,
        "verdict": "GO" if all(criteria.values()) else "NO-GO",
    }
    print("\n=== V2 GO / NO-GO ===", flush=True)
    print(json.dumps(results["go_nogo"], indent=2), flush=True)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        json.dump(results, handle, indent=2)
    print("Wrote", output, flush=True)


if __name__ == "__main__":
    main()
