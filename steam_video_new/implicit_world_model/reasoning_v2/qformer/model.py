"""Neural modules for the four-slot dual Q-Former baseline."""

from __future__ import annotations

import math
from typing import Mapping

import torch
from torch import Tensor, nn

from .contracts import SLOT_NAMES


class FourSlotProjector(nn.Module):
    """Project heterogeneous slot vectors into one typed hidden space."""

    def __init__(self, slot_dimensions: Mapping[str, int], hidden_size: int) -> None:
        super().__init__()
        if set(slot_dimensions) != set(SLOT_NAMES):
            raise ValueError(f"slot dimensions must contain exactly {SLOT_NAMES}")
        self.hidden_size = hidden_size
        self.projectors = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.LayerNorm(int(slot_dimensions[name])),
                    nn.Linear(int(slot_dimensions[name]), hidden_size),
                )
                for name in SLOT_NAMES
            }
        )
        self.type_embeddings = nn.Parameter(torch.empty(len(SLOT_NAMES), hidden_size))
        nn.init.normal_(self.type_embeddings, std=0.02)

    def forward(
        self,
        features: Mapping[str, Tensor],
        validity: Tensor,
    ) -> Tensor:
        if set(features) != set(SLOT_NAMES):
            raise ValueError(f"features must contain exactly {SLOT_NAMES}")
        if validity.ndim != 2 or validity.shape[1] != len(SLOT_NAMES):
            raise ValueError("validity must have shape [B, 4]")
        if validity.dtype is not torch.bool:
            raise ValueError("validity must be boolean")
        projected = torch.stack(
            [self.projectors[name](features[name]) for name in SLOT_NAMES], dim=1
        )
        if projected.shape[:2] != validity.shape:
            raise ValueError("feature batch shape disagrees with validity")
        typed = projected + self.type_embeddings.unsqueeze(0)
        return typed.masked_fill(~validity.unsqueeze(-1), 0.0)


class QueryCrossAttentionBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        *,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError("hidden size must be divisible by attention heads")
        self.self_norm = nn.LayerNorm(hidden_size)
        self.self_attention = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True
        )
        self.cross_norm = nn.LayerNorm(hidden_size)
        self.memory_norm = nn.LayerNorm(hidden_size)
        self.cross_attention = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True
        )
        self.mlp_norm = nn.LayerNorm(hidden_size)
        intermediate = round(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, intermediate),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate, hidden_size),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        sequence: Tensor,
        memory: Tensor,
        *,
        sequence_validity: Tensor,
        memory_validity: Tensor,
    ) -> Tensor:
        normalized = self.self_norm(sequence)
        attended, _ = self.self_attention(
            normalized,
            normalized,
            normalized,
            key_padding_mask=~sequence_validity,
            need_weights=False,
        )
        sequence = sequence + attended
        attended, _ = self.cross_attention(
            self.cross_norm(sequence),
            self.memory_norm(memory),
            self.memory_norm(memory),
            key_padding_mask=~memory_validity,
            need_weights=False,
        )
        sequence = sequence + attended
        return sequence + self.mlp(self.mlp_norm(sequence))


class QueryFormer(nn.Module):
    """Small query bottleneck with optional conditioning tokens."""

    def __init__(
        self,
        *,
        hidden_size: int = 768,
        num_queries: int = 8,
        num_layers: int = 2,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if num_queries <= 0 or num_layers <= 0:
            raise ValueError("Q-Former query and layer counts must be positive")
        self.hidden_size = hidden_size
        self.num_queries = num_queries
        self.query_tokens = nn.Parameter(torch.empty(num_queries, hidden_size))
        nn.init.normal_(self.query_tokens, std=0.02)
        self.layers = nn.ModuleList(
            QueryCrossAttentionBlock(
                hidden_size,
                num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
            )
            for _ in range(num_layers)
        )
        self.output_norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        memory: Tensor,
        memory_validity: Tensor,
        *,
        conditioning: Tensor | None = None,
        conditioning_validity: Tensor | None = None,
    ) -> Tensor:
        if memory.ndim != 3 or memory.shape[-1] != self.hidden_size:
            raise ValueError("memory must have shape [B, M, hidden_size]")
        if memory_validity.shape != memory.shape[:2] or memory_validity.dtype is not torch.bool:
            raise ValueError("memory validity must be boolean with shape [B, M]")
        if (~memory_validity.any(dim=1)).any():
            raise ValueError("every Q-Former sample requires at least one valid memory token")
        batch_size = memory.shape[0]
        queries = self.query_tokens.unsqueeze(0).expand(batch_size, -1, -1)
        query_validity = torch.ones(
            batch_size, self.num_queries, dtype=torch.bool, device=memory.device
        )
        if conditioning is None:
            sequence = queries
            sequence_validity = query_validity
        else:
            if conditioning.ndim != 3 or conditioning.shape[::2] != (
                batch_size,
                self.hidden_size,
            ):
                raise ValueError("conditioning must have shape [B, L, hidden_size]")
            if conditioning_validity is None:
                conditioning_validity = torch.ones(
                    conditioning.shape[:2], dtype=torch.bool, device=conditioning.device
                )
            if (
                conditioning_validity.shape != conditioning.shape[:2]
                or conditioning_validity.dtype is not torch.bool
            ):
                raise ValueError("conditioning validity has an invalid shape or dtype")
            sequence = torch.cat((queries, conditioning), dim=1)
            sequence_validity = torch.cat((query_validity, conditioning_validity), dim=1)
        for layer in self.layers:
            sequence = layer(
                sequence,
                memory,
                sequence_validity=sequence_validity,
                memory_validity=memory_validity,
            )
        return self.output_norm(sequence[:, : self.num_queries])


class IndependentNodeQFormer(QueryFormer):
    """QF1: question-independent fusion of four typed feature slots."""

    def __init__(self, **kwargs: object) -> None:
        kwargs.setdefault("num_queries", 8)
        kwargs.setdefault("num_layers", 4)
        super().__init__(**kwargs)


class ConditionedProposalQFormer(QueryFormer):
    """QF2: question-conditioned proposal tokens over frozen QF1 slots."""

    def __init__(self, **kwargs: object) -> None:
        kwargs.setdefault("num_queries", 8)
        kwargs.setdefault("num_layers", 2)
        super().__init__(**kwargs)

    def forward(
        self,
        qf1_tokens: Tensor,
        question_tokens: Tensor,
        *,
        qf1_validity: Tensor | None = None,
        question_validity: Tensor | None = None,
    ) -> Tensor:
        if qf1_validity is None:
            qf1_validity = torch.ones(
                qf1_tokens.shape[:2], dtype=torch.bool, device=qf1_tokens.device
            )
        return super().forward(
            qf1_tokens,
            qf1_validity,
            conditioning=question_tokens,
            conditioning_validity=question_validity,
        )


class FourSlotDecoder(QueryFormer):
    """Training-only decoder that recovers the four projected input slots."""

    def __init__(self, **kwargs: object) -> None:
        kwargs.setdefault("num_queries", len(SLOT_NAMES))
        kwargs.setdefault("num_layers", 1)
        super().__init__(**kwargs)

    def forward(self, qf1_tokens: Tensor) -> Tensor:
        validity = torch.ones(
            qf1_tokens.shape[:2], dtype=torch.bool, device=qf1_tokens.device
        )
        return super().forward(qf1_tokens, validity)


class HeterogeneousSlotHeads(nn.Module):
    """Map four decoded hidden slots back to their frozen source dimensions."""

    def __init__(self, slot_dimensions: Mapping[str, int], hidden_size: int) -> None:
        super().__init__()
        if set(slot_dimensions) != set(SLOT_NAMES):
            raise ValueError(f"slot dimensions must contain exactly {SLOT_NAMES}")
        self.heads = nn.ModuleDict(
            {
                name: nn.Linear(hidden_size, int(slot_dimensions[name]))
                for name in SLOT_NAMES
            }
        )

    def forward(self, decoded_slots: Tensor) -> dict[str, Tensor]:
        if decoded_slots.ndim != 3 or decoded_slots.shape[1] != len(SLOT_NAMES):
            raise ValueError("decoded slots must have shape [B, 4, hidden_size]")
        return {
            name: self.heads[name](decoded_slots[:, index])
            for index, name in enumerate(SLOT_NAMES)
        }


class NodeScoreHead(nn.Module):
    """Aggregate token-question similarities with a stable log-mean-exp."""

    def __init__(self, hidden_size: int, *, query_temperature: float = 0.07) -> None:
        super().__init__()
        if query_temperature <= 0:
            raise ValueError("query temperature must be positive")
        self.proposal_projection = nn.Linear(hidden_size, hidden_size, bias=False)
        self.question_projection = nn.Linear(hidden_size, hidden_size, bias=False)
        self.query_temperature = query_temperature

    def forward(
        self,
        proposals: Tensor,
        question_tokens: Tensor,
        question_validity: Tensor | None = None,
    ) -> Tensor:
        single_node = proposals.ndim == 3
        if single_node:
            proposals = proposals.unsqueeze(1)
        if proposals.ndim != 4 or question_tokens.ndim != 3:
            raise ValueError("expected proposals [B,N,Q,D] and questions [B,L,D]")
        if proposals.shape[0] != question_tokens.shape[0]:
            raise ValueError("proposal and question batch sizes differ")
        if question_validity is None:
            question_validity = torch.ones(
                question_tokens.shape[:2],
                dtype=torch.bool,
                device=question_tokens.device,
            )
        weights = question_validity.to(question_tokens.dtype).unsqueeze(-1)
        denominator = weights.sum(dim=1).clamp_min(1.0)
        pooled_question = (question_tokens * weights).sum(dim=1) / denominator
        proposal = nn.functional.normalize(
            self.proposal_projection(proposals), dim=-1
        )
        question = nn.functional.normalize(
            self.question_projection(pooled_question), dim=-1
        )
        similarities = torch.einsum("bnqd,bd->bnq", proposal, question)
        temperature = self.query_temperature
        scores = temperature * (
            torch.logsumexp(similarities / temperature, dim=-1)
            - math.log(similarities.shape[-1])
        )
        return scores[:, 0] if single_node else scores
