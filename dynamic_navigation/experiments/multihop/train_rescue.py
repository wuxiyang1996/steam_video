#!/usr/bin/env python3
"""Route-A rescue judgment: synthetic multi-hop + weak L_action + action-LoRA.

Go criteria (all required):
  1) discrete_action acc > one_shot acc
  2) shuffle drops >= 0.02 absolute (2 points)
  3) m_k rises (m_gain > 0 and mono_rate > 0.5)
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

from encoder import build_encoder
from metrics import pack_results
from model import (
    ACTIONS,
    ACTION2ID,
    DiscreteActionNavigator,
    OneShotReader,
    PlainRecurrent,
    action_sketch_loss,
    complementarity_loss,
    entropy_bonus,
    progress_loss,
)
from synthetic import load_synthetic_sets
from train_eval import CachedMH, collate, precompute


def train_model(model, loader, device, epochs, k_steps, lr, lambda_a_warmup=0.05, warmup_epochs=2):
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    for ep in range(epochs):
        model.train()
        # weak / warmup-only action imitation
        la_w = lambda_a_warmup if ep < warmup_epochs else 0.0
        total, n = 0.0, 0
        for batch in loader:
            mem = batch["mem"].to(device)
            q = batch["q"].to(device)
            cands = batch["cands"].to(device)
            out = model(mem, q, cands)
            ce = F.cross_entropy(out["logits"], torch.zeros(mem.size(0), dtype=torch.long, device=device))
            lp = progress_loss(out["margins"])
            lc = complementarity_loss(out.get("alphas"))
            if not torch.is_tensor(lc):
                lc = torch.tensor(0.0, device=device)
            lc = lc.to(device)
            la = action_sketch_loss(out.get("pis"), batch["sketch"], k_steps)
            if not torch.is_tensor(la):
                la = torch.tensor(0.0, device=device)
            la = la.to(device)
            ent = entropy_bonus(out.get("pis"))
            if not torch.is_tensor(ent):
                ent = torch.tensor(0.0, device=device)
            ent = ent.to(device)
            # maximize entropy => subtract
            loss = ce + 0.5 * lp + 0.05 * lc + la_w * la - 0.05 * ent
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * mem.size(0)
            n += mem.size(0)
        print(f"  epoch {ep+1}/{epochs} loss={total/max(n,1):.4f} lambda_a={la_w}", flush=True)
    return model


@torch.no_grad()
def evaluate(model, loader, device, name, **kwargs):
    model.eval()
    logits, margins, types, acts = [], [], [], []
    for batch in loader:
        out = model(batch["mem"].to(device), batch["q"].to(device), batch["cands"].to(device), **kwargs)
        logits.append(out["logits"].cpu())
        margins.append(out["margins"].cpu())
        types.extend(batch["qtype"])
        if out.get("actions") is not None:
            acts.append(out["actions"].cpu())
    res = pack_results(name, torch.cat(logits), torch.cat(margins), types)
    if acts:
        a = torch.cat(acts)
        res["action_freq"] = {ACTIONS[i]: float((a == i).float().mean()) for i in range(len(ACTIONS))}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="qwen3-emb-0.6b")
    ap.add_argument("--n-train", type=int, default=2048)
    ap.add_argument("--n-val", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--n-tokens", type=int, default=16)
    ap.add_argument("--k-steps", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="outputs/rescue_synth_go_nogo.json")
    ap.add_argument("--hf-home", default="/fs/gamma-projects/vlm-robot/hf_cache")
    args = ap.parse_args()

    os.environ.setdefault("HF_HOME", args.hf_home)
    os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(args.hf_home, "datasets"))
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} encoder={args.encoder}", flush=True)
    if device == "cuda":
        print("gpu=", torch.cuda.get_device_name(0), flush=True)

    train_ex, val_ex = load_synthetic_sets(args.n_train, args.n_val, seed=args.seed)
    print(f"synthetic train={len(train_ex)} val={len(val_ex)} cands={len(train_ex[0].candidates)}", flush=True)
    print("example Q:", val_ex[0].question, flush=True)
    print("example A:", val_ex[0].answer, flush=True)
    print("example ctx:\n", val_ex[0].context, flush=True)

    enc = build_encoder(args.encoder, device=device)
    print("encoding...", flush=True)
    tr_m, tr_q, tr_c = precompute(enc, train_ex, args.n_tokens, device)
    va_m, va_q, va_c = precompute(enc, val_ex, args.n_tokens, device)
    tr_m, tr_q, tr_c = F.normalize(tr_m, dim=-1), F.normalize(tr_q, dim=-1), F.normalize(tr_c, dim=-1)
    va_m, va_q, va_c = F.normalize(va_m, dim=-1), F.normalize(va_q, dim=-1), F.normalize(va_c, dim=-1)
    del enc
    if device == "cuda":
        torch.cuda.empty_cache()

    train_loader = DataLoader(CachedMH(train_ex, tr_m, tr_q, tr_c), batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(CachedMH(val_ex, va_m, va_q, va_c), batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    in_dim = tr_m.size(-1)

    results = {"config": vars(args), "models": [], "interventions": [], "go_nogo": {}}

    # baselines
    for name, factory in [
        ("one_shot", lambda: OneShotReader(in_dim, args.d_model, args.n_tokens)),
        ("plain_recurrent", lambda: PlainRecurrent(in_dim, args.d_model, args.n_tokens, args.k_steps)),
    ]:
        print(f"\n=== {name} ===", flush=True)
        model = factory().to(device)
        train_model(model, train_loader, device, args.epochs, args.k_steps, args.lr, lambda_a_warmup=0.0, warmup_epochs=0)
        res = evaluate(model, val_loader, device, name)
        print(json.dumps(res, indent=2), flush=True)
        results["models"].append(res)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    print("\n=== discrete_action (LoRA, weak L_action) ===", flush=True)
    discrete = DiscreteActionNavigator(
        in_dim, args.d_model, args.n_tokens, k_steps=args.k_steps, lora_rank=16
    ).to(device)
    train_model(discrete, train_loader, device, args.epochs, args.k_steps, args.lr, lambda_a_warmup=0.05, warmup_epochs=2)
    res_d = evaluate(discrete, val_loader, device, "discrete_action")
    print(json.dumps(res_d, indent=2), flush=True)
    results["models"].append(res_d)

    # interventions: temporal shuffle of predicted action sequences
    @torch.no_grad()
    def eval_temporal_shuffle():
        discrete.eval()
        # first get predicted actions
        pred_actions = []
        for batch in val_loader:
            out = discrete(batch["mem"].to(device), batch["q"].to(device), batch["cands"].to(device))
            pred_actions.append(out["actions"].cpu())
        pred_actions = torch.cat(pred_actions, dim=0)  # [N,K]
        # permute steps for each example
        shuffled = pred_actions.clone()
        for i in range(shuffled.size(0)):
            perm = torch.randperm(shuffled.size(1))
            shuffled[i] = shuffled[i, perm]

        logits, margins, types = [], [], []
        offset = 0
        for batch in val_loader:
            bsz = batch["mem"].size(0)
            force = shuffled[offset : offset + bsz].to(device)
            offset += bsz
            out = discrete(
                batch["mem"].to(device),
                batch["q"].to(device),
                batch["cands"].to(device),
                force_actions=force,
            )
            logits.append(out["logits"].cpu())
            margins.append(out["margins"].cpu())
            types.extend(batch["qtype"])
        return pack_results("discrete_action/shuffle_temporal", torch.cat(logits), torch.cat(margins), types)

    res_s = eval_temporal_shuffle()
    print(json.dumps(res_s, indent=2), flush=True)
    results["interventions"].append(res_s)

    # also random action sequences
    @torch.no_grad()
    def eval_random_actions():
        discrete.eval()
        logits, margins, types = [], [], []
        for batch in val_loader:
            bsz = batch["mem"].size(0)
            force = torch.randint(0, len(ACTIONS), (bsz, args.k_steps), device=device)
            out = discrete(
                batch["mem"].to(device),
                batch["q"].to(device),
                batch["cands"].to(device),
                force_actions=force,
            )
            logits.append(out["logits"].cpu())
            margins.append(out["margins"].cpu())
            types.extend(batch["qtype"])
        return pack_results("discrete_action/random_actions", torch.cat(logits), torch.cat(margins), types)

    res_r = eval_random_actions()
    print(json.dumps(res_r, indent=2), flush=True)
    results["interventions"].append(res_r)

    for aname in ["GROUND", "TRACE", "COMPOSE"]:
        aid = ACTION2ID[aname]

        @torch.no_grad()
        def force_eval(act=aid, name=aname):
            discrete.eval()
            logits, margins, types = [], [], []
            for batch in val_loader:
                force = torch.full((batch["mem"].size(0), args.k_steps), act, device=device, dtype=torch.long)
                out = discrete(batch["mem"].to(device), batch["q"].to(device), batch["cands"].to(device), force_actions=force)
                logits.append(out["logits"].cpu())
                margins.append(out["margins"].cpu())
                types.extend(batch["qtype"])
            return pack_results(f"discrete_action/force_{name}", torch.cat(logits), torch.cat(margins), types)

        r = force_eval()
        print(json.dumps(r, indent=2), flush=True)
        results["interventions"].append(r)

    # Go / No-Go
    by_name = {m["model"]: m for m in results["models"]}
    one = by_name["one_shot"]["acc"]
    disc = by_name["discrete_action"]["acc"]
    shuf = res_s["acc"]
    drop = disc - shuf
    m_gain = by_name["discrete_action"]["m_gain"]
    mono = by_name["discrete_action"]["mono_rate"]
    go1 = disc > one
    go2 = drop >= 0.02
    go3 = (m_gain > 0) and (mono > 0.5)
    verdict = "GO" if (go1 and go2 and go3) else "NO-GO"
    results["go_nogo"] = {
        "discrete_acc": disc,
        "one_shot_acc": one,
        "shuffle_acc": shuf,
        "shuffle_drop": drop,
        "m_gain": m_gain,
        "mono_rate": mono,
        "criteria": {
            "discrete_gt_oneshot": go1,
            "shuffle_drop_ge_2pts": go2,
            "m_k_rises": go3,
        },
        "verdict": verdict,
    }
    print("\n==== GO / NO-GO ====", flush=True)
    print(json.dumps(results["go_nogo"], indent=2), flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print("Wrote", out, flush=True)


if __name__ == "__main__":
    main()
