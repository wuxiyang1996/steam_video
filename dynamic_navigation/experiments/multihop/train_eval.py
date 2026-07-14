#!/usr/bin/env python3
"""Week 0.5 HotpotQA soft-mixture navigator experiments."""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from data import MHExample, load_hotpot_subset
from encoder import build_encoder
from metrics import pack_results
from model import (
    ACTIONS,
    ACTION2ID,
    DiscreteActionNavigator,
    OneShotReader,
    PlainRecurrent,
    SoftMixtureNavigator,
    action_sketch_loss,
    complementarity_loss,
    progress_loss,
)


class CachedMH(Dataset):
    def __init__(self, examples: list[MHExample], mem, q, cands):
        self.examples = examples
        self.mem = mem  # [N,P,d]
        self.q = q
        self.cands = cands

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return {
            "mem": self.mem[i],
            "q": self.q[i],
            "cands": self.cands[i],
            "sketch": self.examples[i].action_sketch,
            "qtype": self.examples[i].qtype,
            "answer": self.examples[i].answer,
        }


def collate(batch):
    return {
        "mem": torch.stack([b["mem"] for b in batch], dim=0),
        "q": torch.stack([b["q"] for b in batch], dim=0),
        "cands": torch.stack([b["cands"] for b in batch], dim=0),
        "sketch": [b["sketch"] for b in batch],
        "qtype": [b["qtype"] for b in batch],
        "answer": [b["answer"] for b in batch],
    }


@torch.no_grad()
def precompute(encoder, examples: list[MHExample], n_tokens: int, device: str):
    mems, qs, cands = [], [], []
    for i, ex in enumerate(examples):
        m = encoder.encode_paragraphs_as_memory(ex.context, n_tokens=n_tokens)
        q = encoder.encode([ex.question])[0]
        c = encoder.encode(ex.candidates)
        mems.append(m.cpu())
        qs.append(q.cpu())
        cands.append(c.cpu())
        if (i + 1) % 50 == 0:
            print(f"  encoded {i+1}/{len(examples)}", flush=True)
    return torch.stack(mems), torch.stack(qs), torch.stack(cands)


def train_one(model, loader, opt, device, epochs, k_steps, lambdas):
    model.train()
    for ep in range(epochs):
        total = 0.0
        n = 0
        for batch in loader:
            mem = batch["mem"].to(device)
            q = batch["q"].to(device)
            cands = batch["cands"].to(device)
            out = model(mem, q, cands)
            ce = F.cross_entropy(out["logits"], torch.zeros(mem.size(0), dtype=torch.long, device=device))
            lp = progress_loss(out["margins"])
            lc = complementarity_loss(out.get("alphas"))
            if not torch.is_tensor(lc):
                lc = torch.tensor(lc, device=device)
            la = action_sketch_loss(out.get("pis"), batch["sketch"], k_steps)
            if not torch.is_tensor(la):
                la = torch.tensor(la, device=device)
            la = la.to(device)
            lc = lc.to(device)
            loss = ce + lambdas["p"] * lp + lambdas["c"] * lc + lambdas["a"] * la
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * mem.size(0)
            n += mem.size(0)
        print(f"  epoch {ep+1}/{epochs} loss={total/max(n,1):.4f}", flush=True)


@torch.no_grad()
def evaluate(model, loader, device, name: str, **fwd_kwargs):
    model.eval()
    all_logits, all_margins, all_types, all_actions = [], [], [], []
    for batch in loader:
        out = model(
            batch["mem"].to(device),
            batch["q"].to(device),
            batch["cands"].to(device),
            **fwd_kwargs,
        )
        all_logits.append(out["logits"].cpu())
        all_margins.append(out["margins"].cpu())
        all_types.extend(batch["qtype"])
        if out.get("actions") is not None:
            all_actions.append(out["actions"].cpu())
    logits = torch.cat(all_logits, dim=0)
    margins = torch.cat(all_margins, dim=0)
    res = pack_results(name, logits, margins, all_types)
    if all_actions:
        acts = torch.cat(all_actions, dim=0)
        # action histogram over steps
        hist = {}
        for i, a in enumerate(ACTIONS):
            hist[a] = float((acts == i).float().mean().item())
        res["action_freq"] = hist
    return res


def sketch_tensor(sketches: list[list[str]], k_steps: int, device) -> torch.Tensor:
    rows = []
    for sk in sketches:
        ids = [ACTION2ID.get(a, ACTION2ID["COMPOSE"]) for a in sk[:k_steps]]
        while len(ids) < k_steps:
            ids.append(ACTION2ID["STOP"])
        rows.append(ids[:k_steps])
    return torch.tensor(rows, device=device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="qwen3-emb-0.6b", choices=["minilm", "bert", "qwen3-emb-0.6b", "qwen3-emb-2b"])
    ap.add_argument("--policy", default="mlp", choices=["mlp", "bert", "qwen3.5-2b"])
    ap.add_argument("--n-train", type=int, default=512)
    ap.add_argument("--n-val", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--n-tokens", type=int, default=16)
    ap.add_argument("--k-steps", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="outputs/discrete_action_results.json")
    ap.add_argument("--hf-home", type=str, default="/fs/gamma-projects/vlm-robot/hf_cache")
    ap.add_argument("--skip-baselines", action="store_true")
    args = ap.parse_args()

    os.environ.setdefault("HF_HOME", args.hf_home)
    os.environ.setdefault("TRANSFORMERS_CACHE", args.hf_home)
    os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(args.hf_home, "datasets"))

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} encoder={args.encoder} policy={args.policy}", flush=True)
    if device == "cuda":
        print(f"gpu={torch.cuda.get_device_name(0)}", flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("Loading HotpotQA distractor...", flush=True)
    train_ex = load_hotpot_subset("train", n=args.n_train, seed=args.seed)
    val_ex = load_hotpot_subset("validation", n=args.n_val, seed=args.seed + 1)
    print(f"train={len(train_ex)} val={len(val_ex)}", flush=True)

    print(f"Building frozen encoder {args.encoder}...", flush=True)
    t0 = time.time()
    encoder = build_encoder(args.encoder, device=device)
    print(f"encoder ready in {time.time()-t0:.1f}s dim={encoder.out_dim}", flush=True)

    print("Precomputing embeddings (train)...", flush=True)
    tr_mem, tr_q, tr_c = precompute(encoder, train_ex, args.n_tokens, device)
    print("Precomputing embeddings (val)...", flush=True)
    va_mem, va_q, va_c = precompute(encoder, val_ex, args.n_tokens, device)

    tr_mem = F.normalize(tr_mem, dim=-1)
    tr_q = F.normalize(tr_q, dim=-1)
    tr_c = F.normalize(tr_c, dim=-1)
    va_mem = F.normalize(va_mem, dim=-1)
    va_q = F.normalize(va_q, dim=-1)
    va_c = F.normalize(va_c, dim=-1)

    del encoder
    if device == "cuda":
        torch.cuda.empty_cache()

    train_ds = CachedMH(train_ex, tr_mem, tr_q, tr_c)
    val_ds = CachedMH(val_ex, va_mem, va_q, va_c)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    in_dim = tr_mem.size(-1)
    lambdas = {"p": 0.5, "c": 0.1, "a": 0.3}
    results = {
        "config": vars(args),
        "device": device,
        "in_dim": in_dim,
        "note": (
            "Discrete action policy + FiLM read + progress reward on HotpotQA distractor. "
            "Policy conditioned on (u,r,a_prev,q). Train with Gumbel-Softmax hard; test argmax. "
            f"policy={args.policy} (MLP head; BERT/Qwen3.5-2B reserved for heavier policy ablations)."
        ),
        "models": [],
        "interventions": [],
    }

    specs = []
    if not args.skip_baselines:
        specs.extend([
            ("one_shot", lambda: OneShotReader(in_dim, args.d_model, args.n_tokens)),
            ("plain_recurrent", lambda: PlainRecurrent(in_dim, args.d_model, args.n_tokens, args.k_steps)),
            ("soft_mixture", lambda: SoftMixtureNavigator(in_dim, args.d_model, args.n_tokens, k_steps=args.k_steps, policy=args.policy)),
        ])
    specs.append(
        ("discrete_action", lambda: DiscreteActionNavigator(
            in_dim, args.d_model, args.n_tokens, k_steps=args.k_steps, policy=args.policy
        ))
    )

    discrete_model = None
    for name, factory in specs:
        print(f"\n=== training {name} ===", flush=True)
        model = factory().to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
        train_one(model, train_loader, opt, device, args.epochs, args.k_steps, lambdas)
        res = evaluate(model, val_loader, device, name)
        print(json.dumps(res, indent=2), flush=True)
        results["models"].append(res)
        if name == "discrete_action":
            discrete_model = model
        else:
            del model
            if device == "cuda":
                torch.cuda.empty_cache()

    # Interventions on discrete action policy
    if discrete_model is not None:
        print("\n=== interventions on discrete_action ===", flush=True)
        for tag, kwargs in [
            ("shuffle_actions", {"shuffle_actions": True}),
        ]:
            res = evaluate(discrete_model, val_loader, device, f"discrete_action/{tag}", **kwargs)
            print(json.dumps(res, indent=2), flush=True)
            results["interventions"].append(res)

        # force same action every step
        for act_name in ["GROUND", "TRACE", "COMPOSE"]:
            aid = ACTION2ID[act_name]

            @torch.no_grad()
            def eval_force(loader=val_loader, act=aid, aname=act_name):
                discrete_model.eval()
                all_logits, all_margins, all_types = [], [], []
                for batch in loader:
                    bsz = batch["mem"].size(0)
                    force = torch.full((bsz, args.k_steps), act, device=device, dtype=torch.long)
                    out = discrete_model(
                        batch["mem"].to(device),
                        batch["q"].to(device),
                        batch["cands"].to(device),
                        force_actions=force,
                    )
                    all_logits.append(out["logits"].cpu())
                    all_margins.append(out["margins"].cpu())
                    all_types.extend(batch["qtype"])
                return pack_results(f"discrete_action/force_{aname}", torch.cat(all_logits), torch.cat(all_margins), all_types)

            res = eval_force()
            print(json.dumps(res, indent=2), flush=True)
            results["interventions"].append(res)

        # teacher sketch force (upper bound-ish)
        @torch.no_grad()
        def eval_teacher():
            discrete_model.eval()
            all_logits, all_margins, all_types = [], [], []
            for batch in val_loader:
                force = sketch_tensor(batch["sketch"], args.k_steps, device)
                out = discrete_model(
                    batch["mem"].to(device),
                    batch["q"].to(device),
                    batch["cands"].to(device),
                    force_actions=force,
                )
                all_logits.append(out["logits"].cpu())
                all_margins.append(out["margins"].cpu())
                all_types.extend(batch["qtype"])
            return pack_results("discrete_action/force_teacher_sketch", torch.cat(all_logits), torch.cat(all_margins), all_types)

        res = eval_teacher()
        print(json.dumps(res, indent=2), flush=True)
        results["interventions"].append(res)

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
