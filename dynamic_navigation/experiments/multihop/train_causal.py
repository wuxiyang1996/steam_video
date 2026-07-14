#!/usr/bin/env python3
"""Counterfactual causal navigation experiment.

The navigator chooses a continuous direction from the previous state, read,
top-2 answer contrast, and previous causal gain. Each read is validated with
do(r_k=0), then accepted through a learned causal-gain gate.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data import load_hotpot_subset
from encoder import build_encoder
from metrics import pack_results
from model import (
    CounterfactualCausalNavigator,
    OneShotReader,
    PlainRecurrent,
    causal_necessity_loss,
    direction_novelty_loss,
    progress_loss,
)
from synthetic import load_synthetic_sets
from train_eval import CachedMH, collate, precompute


def train_model(model, loader, device, epochs, lr, causal: bool):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        totals = {"loss": 0.0, "qa": 0.0, "progress": 0.0, "causal": 0.0}
        count = 0
        for batch in loader:
            memory = batch["mem"].to(device)
            question = batch["q"].to(device)
            candidates = batch["cands"].to(device)
            output = model(memory, question, candidates)
            target = torch.zeros(memory.size(0), dtype=torch.long, device=device)
            qa = F.cross_entropy(output["logits"], target)
            progress = progress_loss(output["margins"])
            causal_loss = causal_necessity_loss(output.get("causal_gains")).to(device)
            novelty = direction_novelty_loss(output.get("directions")).to(device)
            if causal:
                loss = qa + 0.5 * progress + 0.5 * causal_loss + 0.05 * novelty
            else:
                loss = qa + 0.5 * progress
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            batch_size = memory.size(0)
            totals["loss"] += loss.item() * batch_size
            totals["qa"] += qa.item() * batch_size
            totals["progress"] += progress.item() * batch_size
            totals["causal"] += causal_loss.item() * batch_size
            count += batch_size
        metrics = " ".join(f"{k}={v / count:.4f}" for k, v in totals.items())
        print(f"  epoch {epoch + 1}/{epochs} {metrics}", flush=True)


@torch.no_grad()
def evaluate(model, loader, device, name: str, intervention: str | None = None):
    model.eval()
    logits, margins, qtypes = [], [], []
    gains, gates = [], []
    for batch in loader:
        kwargs = {}
        if isinstance(model, CounterfactualCausalNavigator):
            kwargs["intervention"] = intervention
        output = model(
            batch["mem"].to(device),
            batch["q"].to(device),
            batch["cands"].to(device),
            **kwargs,
        )
        logits.append(output["logits"].cpu())
        margins.append(output["margins"].cpu())
        qtypes.extend(batch["qtype"])
        if output.get("causal_gains") is not None:
            gains.append(output["causal_gains"].cpu())
            gates.append(output["gates"].cpu())
    result = pack_results(name, torch.cat(logits), torch.cat(margins), qtypes)
    if gains:
        gain = torch.cat(gains)
        gate = torch.cat(gates)
        result["causal_gain_mean"] = float(gain.mean())
        result["causal_positive_rate"] = float((gain > 0).float().mean())
        result["accept_gate_mean"] = float(gate.mean())
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["hotpot", "synthetic"], default="hotpot")
    parser.add_argument("--encoder", default="qwen3-emb-0.6b")
    parser.add_argument("--n-train", type=int, default=2048)
    parser.add_argument("--n-val", type=int, default=512)
    parser.add_argument("--n-distractors", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-tokens", type=int, default=16)
    parser.add_argument("--k-steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="outputs/causal_navigation.json")
    parser.add_argument(
        "--hf-home", default="/fs/gamma-projects/vlm-robot/hf_cache"
    )
    args = parser.parse_args()

    os.environ.setdefault("HF_HOME", args.hf_home)
    os.environ.setdefault(
        "HF_DATASETS_CACHE", os.path.join(args.hf_home, "datasets")
    )
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} dataset={args.dataset} encoder={args.encoder}", flush=True)
    if device == "cuda":
        print("gpu=", torch.cuda.get_device_name(0), flush=True)

    if args.dataset == "hotpot":
        train_examples = load_hotpot_subset(
            "train", args.n_train, args.seed, args.n_distractors
        )
        val_examples = load_hotpot_subset(
            "validation", args.n_val, args.seed + 1, args.n_distractors
        )
    else:
        train_examples, val_examples = load_synthetic_sets(
            args.n_train, args.n_val, args.seed
        )
    print(
        f"train={len(train_examples)} val={len(val_examples)} "
        f"candidates={len(train_examples[0].candidates)}",
        flush=True,
    )

    encoder = build_encoder(args.encoder, device=device)
    print("precomputing frozen embeddings...", flush=True)
    tr_m, tr_q, tr_c = precompute(
        encoder, train_examples, args.n_tokens, device
    )
    va_m, va_q, va_c = precompute(encoder, val_examples, args.n_tokens, device)
    tr_m, tr_q, tr_c = (
        F.normalize(tr_m, dim=-1),
        F.normalize(tr_q, dim=-1),
        F.normalize(tr_c, dim=-1),
    )
    va_m, va_q, va_c = (
        F.normalize(va_m, dim=-1),
        F.normalize(va_q, dim=-1),
        F.normalize(va_c, dim=-1),
    )
    del encoder
    if device == "cuda":
        torch.cuda.empty_cache()

    train_loader = DataLoader(
        CachedMH(train_examples, tr_m, tr_q, tr_c),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
    )
    val_loader = DataLoader(
        CachedMH(val_examples, va_m, va_q, va_c),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate,
    )
    input_dim = tr_m.size(-1)
    results = {"config": vars(args), "models": [], "interventions": []}

    specifications = [
        (
            "one_shot",
            lambda: OneShotReader(input_dim, args.d_model, args.n_tokens),
            False,
        ),
        (
            "plain_recurrent",
            lambda: PlainRecurrent(
                input_dim, args.d_model, args.n_tokens, args.k_steps
            ),
            False,
        ),
        (
            "causal_navigation",
            lambda: CounterfactualCausalNavigator(
                input_dim,
                args.d_model,
                args.n_tokens,
                args.k_steps,
            ),
            True,
        ),
    ]

    causal_model = None
    for name, factory, is_causal in specifications:
        print(f"\n=== {name} ===", flush=True)
        model = factory().to(device)
        train_model(model, train_loader, device, args.epochs, args.lr, is_causal)
        result = evaluate(model, val_loader, device, name)
        print(json.dumps(result, indent=2), flush=True)
        results["models"].append(result)
        if is_causal:
            causal_model = model
        else:
            del model
            if device == "cuda":
                torch.cuda.empty_cache()

    assert causal_model is not None
    for intervention in [
        "zero_read",
        "random_direction",
        "reject_all",
        "accept_all",
    ]:
        result = evaluate(
            causal_model,
            val_loader,
            device,
            f"causal_navigation/{intervention}",
            intervention,
        )
        print(json.dumps(result, indent=2), flush=True)
        results["interventions"].append(result)

    by_name = {item["model"]: item for item in results["models"]}
    causal = by_name["causal_navigation"]
    one_shot = by_name["one_shot"]
    zero_read = next(
        item
        for item in results["interventions"]
        if item["model"].endswith("zero_read")
    )
    causal_drop = causal["acc"] - zero_read["acc"]
    criteria = {
        "causal_gt_one_shot": causal["acc"] > one_shot["acc"],
        "do_removal_drop_ge_2pts": causal_drop >= 0.02,
        "m_k_rises": causal["m_gain"] > 0 and causal["mono_rate"] > 0.5,
    }
    results["go_nogo"] = {
        "causal_acc": causal["acc"],
        "one_shot_acc": one_shot["acc"],
        "zero_read_acc": zero_read["acc"],
        "do_removal_drop": causal_drop,
        "criteria": criteria,
        "verdict": "GO" if all(criteria.values()) else "NO-GO",
    }
    print("\n=== CAUSAL GO / NO-GO ===", flush=True)
    print(json.dumps(results["go_nogo"], indent=2), flush=True)

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        json.dump(results, handle, indent=2)
    print("Wrote", output_path, flush=True)


if __name__ == "__main__":
    main()
