"""GPU/CPU forward-backward smoke for separately trained QF1 and QF2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from .contracts import SLOT_NAMES
from .losses import (
    masked_heterogeneous_slot_distillation_loss,
    trusted_multi_positive_retrieval_loss,
)
from .model import (
    ConditionedProposalQFormer,
    FourSlotDecoder,
    FourSlotProjector,
    HeterogeneousSlotHeads,
    IndependentNodeQFormer,
    NodeScoreHead,
)


def run_synthetic_smoke(
    *,
    device: str,
    seed: int = 7,
    qf1_steps: int = 80,
    qf2_steps: int = 80,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    target_device = torch.device(device)
    hidden = 32
    dimensions = {"caption": 12, "entity_state": 10, "visual": 14, "time": 4}
    projector = FourSlotProjector(dimensions, hidden).to(target_device)
    qf1 = IndependentNodeQFormer(
        hidden_size=hidden, num_layers=2, num_heads=4, mlp_ratio=2.0
    ).to(target_device)
    decoder = FourSlotDecoder(
        hidden_size=hidden, num_layers=1, num_heads=4, mlp_ratio=2.0
    ).to(target_device)
    slot_heads = HeterogeneousSlotHeads(dimensions, hidden).to(target_device)
    qf1_optimizer = torch.optim.AdamW(
        [
            *projector.parameters(),
            *qf1.parameters(),
            *decoder.parameters(),
            *slot_heads.parameters(),
        ],
        lr=3e-3,
    )

    batch_size = 24
    features = {
        name: torch.randn(batch_size, dimension, device=target_device)
        for name, dimension in dimensions.items()
    }
    validity = torch.rand(batch_size, len(SLOT_NAMES), device=target_device) > 0.15
    validity[:, 0] = True
    qf1_losses: list[float] = []
    for _ in range(qf1_steps):
        qf1_optimizer.zero_grad(set_to_none=True)
        typed = projector(features, validity)
        u = qf1(typed, validity)
        reconstructed = slot_heads(decoder(u))
        loss = masked_heterogeneous_slot_distillation_loss(
            reconstructed, features, validity
        )
        loss.backward()
        qf1_optimizer.step()
        qf1_losses.append(float(loss.detach()))

    for module in (projector, qf1):
        module.eval()
        for parameter in module.parameters():
            parameter.grad = None
            parameter.requires_grad_(False)

    qf2 = ConditionedProposalQFormer(
        hidden_size=hidden, num_layers=2, num_heads=4, mlp_ratio=2.0
    ).to(target_device)
    scorer = NodeScoreHead(hidden).to(target_device)
    qf2_optimizer = torch.optim.AdamW(
        [*qf2.parameters(), *scorer.parameters()], lr=3e-3
    )
    retrieval_batch = 12
    candidates = 5
    candidate_features = {
        name: torch.randn(
            retrieval_batch * candidates, dimension, device=target_device
        )
        for name, dimension in dimensions.items()
    }
    candidate_validity = torch.ones(
        retrieval_batch * candidates,
        len(SLOT_NAMES),
        dtype=torch.bool,
        device=target_device,
    )
    with torch.no_grad():
        typed_candidates = projector(candidate_features, candidate_validity)
        cached_u = qf1(typed_candidates, candidate_validity)
        question = typed_candidates.reshape(
            retrieval_batch, candidates, len(SLOT_NAMES), hidden
        )[:, 0].mean(dim=1, keepdim=True)
    positive = torch.zeros(
        retrieval_batch, candidates, dtype=torch.bool, device=target_device
    )
    positive[:, 0] = True
    negative = ~positive
    retrieval_losses: list[float] = []
    for _ in range(qf2_steps):
        qf2_optimizer.zero_grad(set_to_none=True)
        repeated_question = question[:, None].expand(-1, candidates, -1, -1).reshape(
            retrieval_batch * candidates, 1, hidden
        )
        proposals = qf2(cached_u, repeated_question).reshape(
            retrieval_batch, candidates, qf2.num_queries, hidden
        )
        scores = scorer(proposals, question)
        loss = trusted_multi_positive_retrieval_loss(scores, positive, negative)
        loss.backward()
        qf2_optimizer.step()
        retrieval_losses.append(float(loss.detach()))

    qf1_gradients_after_freeze = sum(
        parameter.grad is not None and bool(parameter.grad.detach().abs().sum())
        for parameter in qf1.parameters()
    )
    result = {
        "schema_version": "steam-qformer-synthetic-smoke/v0.1",
        "device": str(target_device),
        "seed": seed,
        "qf1": {
            "initial_loss": qf1_losses[0],
            "final_loss": qf1_losses[-1],
            "decreased": qf1_losses[-1] < qf1_losses[0],
            "output_shape": list(u.shape),
        },
        "qf2": {
            "initial_loss": retrieval_losses[0],
            "final_loss": retrieval_losses[-1],
            "decreased": retrieval_losses[-1] < retrieval_losses[0],
            "proposal_shape": list(proposals.shape),
            "score_shape": list(scores.shape),
        },
        "gates": {
            "finite_losses": all(
                torch.isfinite(torch.tensor(qf1_losses + retrieval_losses))
            ),
            "qf1_loss_decreased": qf1_losses[-1] < qf1_losses[0],
            "qf2_loss_decreased": retrieval_losses[-1] < retrieval_losses[0],
            "qf1_frozen_during_qf2": qf1_gradients_after_freeze == 0,
            "independent_query_parameters": qf1.query_tokens.data_ptr()
            != qf2.query_tokens.data_ptr(),
        },
    }
    result["passed"] = all(result["gates"].values())
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--qf1-steps", type=int, default=80)
    parser.add_argument("--qf2-steps", type=int, default=80)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    result = run_synthetic_smoke(
        device=args.device,
        seed=args.seed,
        qf1_steps=args.qf1_steps,
        qf2_steps=args.qf2_steps,
    )
    text = json.dumps(result, indent=2)
    if args.output:
        args.output.expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
        args.output.expanduser().resolve().write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
