"""Token-preserving fixed memory and contrast-conditioned causal navigation v2."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def gather_margin(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    gold = logits.gather(1, targets[:, None]).squeeze(1)
    wrong = logits.masked_fill(
        F.one_hot(targets, logits.size(1)).bool(), float("-inf")
    ).max(dim=1).values
    return gold - wrong


def cosine_logits(
    state: torch.Tensor, candidates: torch.Tensor, temperature: float = 10.0
) -> torch.Tensor:
    """Bounded scoring prevents causal gain from growing by logit scaling."""
    state = F.normalize(state, dim=-1)
    candidates = F.normalize(candidates, dim=-1)
    return temperature * torch.einsum("bd,bcd->bc", state, candidates)


class MaskedPerceiverMemory(nn.Module):
    """Query-independent token-to-latent memory with a real padding mask."""

    def __init__(
        self,
        input_dim: int,
        d_model: int = 256,
        n_latents: int = 32,
        n_heads: int = 8,
    ):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        self.latents = nn.Parameter(torch.randn(n_latents, d_model) * 0.02)
        self.cross_attention = nn.MultiheadAttention(
            d_model, n_heads, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(
        self, token_states: torch.Tensor, token_mask: torch.Tensor
    ) -> torch.Tensor:
        source = self.input_projection(token_states.float())
        latents = self.latents.unsqueeze(0).expand(source.size(0), -1, -1)
        read, _ = self.cross_attention(
            latents,
            source,
            source,
            key_padding_mask=~token_mask.bool(),
            need_weights=False,
        )
        latents = self.norm1(latents + read)
        return self.norm2(latents + self.feed_forward(latents))


class LatentReader(nn.Module):
    """Explicit cosine routing: direction is the only addressing signal."""

    def __init__(self, d_model: int, temperature: float = 0.1):
        super().__init__()
        self.direction_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)
        self.temperature = temperature
        self.output_norm = nn.LayerNorm(d_model)

    def forward(
        self, direction: torch.Tensor, memory: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query = F.normalize(self.direction_projection(direction), dim=-1)
        keys = F.normalize(self.key_projection(memory), dim=-1)
        values = self.value_projection(memory)
        scores = torch.einsum("bd,bnd->bn", query, keys) / self.temperature
        weights = scores.softmax(dim=-1)
        read = torch.einsum("bn,bnd->bd", weights, values)
        return self.output_norm(read), weights


def memory_diversity_loss(memory: torch.Tensor) -> torch.Tensor:
    normalized = F.normalize(memory, dim=-1)
    similarity = torch.matmul(normalized, normalized.transpose(1, 2))
    identity = torch.eye(
        similarity.size(1), device=similarity.device, dtype=torch.bool
    ).unsqueeze(0)
    return similarity.masked_fill(identity, 0).square().mean()


class PooledTokenProbe(nn.Module):
    """Information-retention lower bound using masked mean token states."""

    def __init__(self, input_dim: int, d_model: int = 256):
        super().__init__()
        self.memory_projection = nn.Linear(input_dim, d_model)
        self.question_projection = nn.Linear(input_dim, d_model)
        self.candidate_projection = nn.Linear(input_dim, d_model)
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model), nn.GELU(), nn.LayerNorm(d_model)
        )

    def forward(self, tokens, mask, question, candidates, targets=None, **_):
        weights = mask.unsqueeze(-1).to(tokens.dtype)
        pooled = (tokens * weights).sum(1) / weights.sum(1).clamp_min(1)
        state = self.fusion(
            torch.cat(
                [
                    self.memory_projection(pooled.float()),
                    self.question_projection(question),
                ],
                dim=-1,
            )
        )
        candidate_states = self.candidate_projection(candidates)
        logits = cosine_logits(state, candidate_states)
        margins = (
            gather_margin(logits, targets).unsqueeze(1)
            if targets is not None
            else None
        )
        return {"logits": logits, "margins": margins}


class TokenOneShot(nn.Module):
    """One read over the same fixed Perceiver memory used by the navigator."""

    def __init__(
        self, input_dim: int, d_model: int = 256, n_latents: int = 32
    ):
        super().__init__()
        self.memory_writer = MaskedPerceiverMemory(
            input_dim, d_model, n_latents
        )
        self.question_projection = nn.Linear(input_dim, d_model)
        self.candidate_projection = nn.Linear(input_dim, d_model)
        self.reader = LatentReader(d_model)
        self.update = nn.GRUCell(d_model, d_model)
        self.scorer = nn.Linear(d_model, d_model)

    def forward(self, tokens, mask, question, candidates, targets=None, **_):
        memory = self.memory_writer(tokens, mask)
        query = self.question_projection(question)
        read, attention = self.reader(F.normalize(query, dim=-1), memory)
        state = self.update(read, query)
        candidate_states = self.candidate_projection(candidates)
        logits = cosine_logits(self.scorer(state), candidate_states)
        margins = (
            gather_margin(logits, targets).unsqueeze(1)
            if targets is not None
            else None
        )
        return {
            "logits": logits,
            "margins": margins,
            "alphas": attention.unsqueeze(1),
            "memory_diversity": memory_diversity_loss(memory),
        }


class CausalNavigatorV2(nn.Module):
    """Continuous causal navigation without a query bypass in the reader."""

    def __init__(
        self,
        input_dim: int,
        d_model: int = 256,
        n_latents: int = 32,
        k_steps: int = 4,
        accept_temperature: float = 5.0,
    ):
        super().__init__()
        self.k_steps = k_steps
        self.accept_temperature = accept_temperature
        self.memory_writer = MaskedPerceiverMemory(
            input_dim, d_model, n_latents
        )
        self.question_projection = nn.Linear(input_dim, d_model)
        self.candidate_projection = nn.Linear(input_dim, d_model)
        self.direction_policy = nn.Sequential(
            nn.Linear(d_model * 3 + 1, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.reader = LatentReader(d_model)
        self.update = nn.GRUCell(d_model * 2, d_model)
        self.scorer = nn.Linear(d_model, d_model)

    def _score(self, state, candidates):
        return cosine_logits(self.scorer(state), candidates)

    @staticmethod
    def _contrast(logits, candidates):
        batch = torch.arange(logits.size(0), device=logits.device)
        top2 = logits.topk(2, dim=-1).indices
        first, rival = top2[:, 0], top2[:, 1]
        contrast = candidates[batch, first] - candidates[batch, rival]
        return F.normalize(contrast, dim=-1), first, rival

    @staticmethod
    def _pair_margin(logits, first, rival):
        batch = torch.arange(logits.size(0), device=logits.device)
        return logits[batch, first] - logits[batch, rival]

    def forward(
        self,
        tokens,
        mask,
        question,
        candidates,
        targets=None,
        intervention: str | None = None,
    ):
        memory = self.memory_writer(tokens, mask)
        query = self.question_projection(question)
        candidate_states = self.candidate_projection(candidates)
        state = query
        previous_read = torch.zeros_like(state)
        previous_gain = torch.zeros(
            state.size(0), 1, device=state.device, dtype=state.dtype
        )
        margins, gains, decision_gains, direction_advantages, gates, directions, attentions = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )

        for _ in range(self.k_steps):
            logits_before = self._score(state, candidate_states)
            # Train and test use the same predicted top-2 contrast. Gold labels
            # supervise causal efficacy below, never the navigation input.
            contrast, first, rival = self._contrast(
                logits_before.detach(), candidate_states
            )
            direction = F.normalize(
                self.direction_policy(
                    torch.cat(
                        [state, previous_read, contrast, previous_gain], dim=-1
                    )
                ),
                dim=-1,
            )
            if intervention == "random_direction":
                direction = F.normalize(torch.randn_like(direction), dim=-1)
            elif intervention == "shuffle_direction":
                # In-distribution counterfactual: use another example's valid
                # direction while keeping this example's state and memory.
                direction = direction.roll(shifts=1, dims=0)
            read, attention = self.reader(direction, memory)
            if intervention == "zero_read":
                read = torch.zeros_like(read)

            factual = self.update(
                torch.cat([read, direction], dim=-1), state
            )
            removed = self.update(
                torch.cat([torch.zeros_like(read), direction], dim=-1), state
            )
            factual_logits = self._score(factual, candidate_states)
            removed_logits = self._score(removed, candidate_states)
            decision_gain = self._pair_margin(
                factual_logits, first, rival
            ) - self._pair_margin(removed_logits, first, rival)
            if targets is not None:
                factual_gold_margin = gather_margin(factual_logits, targets)
                removed_gold_margin = gather_margin(removed_logits, targets)
                causal_gain = factual_gold_margin - removed_gold_margin
                # Contrastive intervention: the learned direction must beat a
                # random direction from the same state and memory.
                random_direction = F.normalize(
                    torch.randn_like(direction), dim=-1
                )
                random_read, _ = self.reader(random_direction, memory)
                random_state = self.update(
                    torch.cat([random_read, random_direction], dim=-1), state
                )
                random_logits = self._score(random_state, candidate_states)
                direction_advantage = factual_gold_margin - gather_margin(
                    random_logits, targets
                )
            else:
                causal_gain = decision_gain
                direction_advantage = torch.zeros_like(causal_gain)

            # Keep the main train/test path distribution-matched. A causal gate
            # is evaluated separately only after calibration.
            gate = torch.ones_like(decision_gain).unsqueeze(-1)
            if intervention == "causal_gate":
                gate = torch.sigmoid(
                    self.accept_temperature * decision_gain
                ).unsqueeze(-1)
            if intervention == "reject_all":
                gate = torch.zeros_like(gate)
            elif intervention == "accept_all":
                gate = torch.ones_like(gate)
            state = gate * factual + (1 - gate) * state
            previous_read = gate * read + (1 - gate) * previous_read
            previous_gain = decision_gain.detach().unsqueeze(-1)

            logits = self._score(state, candidate_states)
            if targets is not None:
                margins.append(gather_margin(logits, targets))
            gains.append(causal_gain)
            decision_gains.append(decision_gain)
            direction_advantages.append(direction_advantage)
            gates.append(gate.squeeze(-1))
            directions.append(direction)
            attentions.append(attention)

        return {
            "logits": self._score(state, candidate_states),
            "margins": torch.stack(margins, 1) if margins else None,
            "causal_gains": torch.stack(gains, 1),
            "decision_gains": torch.stack(decision_gains, 1),
            "direction_advantages": torch.stack(direction_advantages, 1),
            "gates": torch.stack(gates, 1),
            "directions": torch.stack(directions, 1),
            "alphas": torch.stack(attentions, 1),
            "memory_diversity": memory_diversity_loss(memory),
        }


def progress_loss(margins: torch.Tensor, delta: float = 0.02):
    if margins is None or margins.size(1) < 2:
        return torch.tensor(0.0, device=margins.device if margins is not None else None)
    return F.relu(margins[:, :-1] + delta - margins[:, 1:]).mean()


def causal_loss(gains: torch.Tensor, gamma: float = 0.02):
    return F.relu(gamma - gains).mean()


def novelty_loss(directions: torch.Tensor):
    if directions.size(1) < 2:
        return directions.new_zeros(())
    similarity = F.cosine_similarity(
        directions[:, 1:], directions[:, :-1], dim=-1
    )
    return similarity.abs().mean()


def direction_advantage_loss(
    advantages: torch.Tensor, gamma: float = 0.05
) -> torch.Tensor:
    return F.relu(gamma - advantages).mean()
