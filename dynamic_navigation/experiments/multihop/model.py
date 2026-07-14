"""Navigators for multi-hop dynamic navigation experiments."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

ACTIONS = ["GROUND", "RESOLVE", "TRACE", "LINK", "CHECK", "COMPOSE", "STOP"]
ACTION2ID = {a: i for i, a in enumerate(ACTIONS)}


class Resampler(nn.Module):
    def __init__(self, in_dim: int, n_tokens: int, d_model: int):
        super().__init__()
        self.n_tokens = n_tokens
        self.proj = nn.Linear(in_dim, d_model)
        self.latents = nn.Parameter(torch.randn(n_tokens, d_model) * 0.02)

    def forward(self, mem: torch.Tensor) -> torch.Tensor:
        x = self.proj(mem)
        if x.size(1) != self.n_tokens:
            x = F.interpolate(
                x.transpose(1, 2), size=self.n_tokens, mode="linear", align_corners=False
            ).transpose(1, 2)
        return x + self.latents.unsqueeze(0)


class CrossRead(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 4):
        super().__init__()
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.out = nn.Linear(d_model, d_model)

    def forward(self, u: torch.Tensor, m: torch.Tensor, e: torch.Tensor | None = None):
        b, n, d = m.shape
        q_in = u if e is None else u + e
        q = self.q(q_in).view(b, 1, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k(m).view(b, n, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v(m).view(b, n, self.n_heads, self.d_head).transpose(1, 2)
        att = torch.matmul(q, k.transpose(-2, -1)) / (self.d_head**0.5)
        alpha = att.softmax(dim=-1)
        r = torch.matmul(alpha, v).transpose(1, 2).contiguous().view(b, d)
        return self.out(r), alpha.mean(dim=1).squeeze(1)


class SoftMixtureNavigator(nn.Module):
    def __init__(self, in_dim, d_model=256, n_tokens=16, n_actions=len(ACTIONS), k_steps=4, policy="mlp"):
        super().__init__()
        self.k_steps = k_steps
        self.n_actions = n_actions
        self.resampler = Resampler(in_dim, n_tokens, d_model)
        self.q_proj = nn.Linear(in_dim, d_model)
        self.a_proj = nn.Linear(in_dim, d_model)
        self.action_emb = nn.Embedding(n_actions, d_model)
        self.policy = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.ReLU(), nn.Linear(d_model, n_actions))
        self.reader = CrossRead(d_model)
        self.gru = nn.GRUCell(d_model * 2, d_model)
        self.score = nn.Linear(d_model, d_model)

    def forward(self, mem_raw, q_emb, cand_emb):
        m = self.resampler(mem_raw)
        u = self.q_proj(q_emb)
        r = torch.zeros_like(u)
        margins, alphas, pi_list = [], [], []
        cand = self.a_proj(cand_emb)
        for _ in range(self.k_steps):
            pi = self.policy(torch.cat([u, r], dim=-1)).softmax(dim=-1)
            e = pi @ self.action_emb.weight
            r, alpha = self.reader(u, m, e)
            u = self.gru(torch.cat([r, e], dim=-1), u)
            s = torch.einsum("bd,bcd->bc", self.score(u), cand)
            margins.append(s[:, 0] - s[:, 1:].max(dim=-1).values)
            alphas.append(alpha)
            pi_list.append(pi)
        return {
            "logits": torch.einsum("bd,bcd->bc", self.score(u), cand),
            "margins": torch.stack(margins, dim=1),
            "alphas": torch.stack(alphas, dim=1),
            "pis": torch.stack(pi_list, dim=1),
            "actions": None,
            "u": u,
        }


class ActionLoRA(nn.Module):
    def __init__(self, d_model: int, n_actions: int, rank: int = 8):
        super().__init__()
        self.A = nn.Parameter(torch.randn(n_actions, rank, d_model) * 0.02)
        self.B = nn.Parameter(torch.zeros(n_actions, d_model, rank))

    def forward(self, x: torch.Tensor, action_ids: torch.Tensor) -> torch.Tensor:
        A = self.A[action_ids]
        B = self.B[action_ids]
        if x.dim() == 2:
            mid = torch.einsum("brd,bd->br", A, x)
            return torch.einsum("bdr,br->bd", B, mid)
        mid = torch.einsum("brd,bnd->bnr", A, x)
        return torch.einsum("bdr,bnr->bnd", B, mid)


class DiscreteActionNavigator(nn.Module):
    """Previous-step-conditioned discrete actions + action-LoRA reads."""

    def __init__(
        self,
        in_dim: int,
        d_model: int = 256,
        n_tokens: int = 16,
        n_actions: int = len(ACTIONS),
        k_steps: int = 4,
        policy: str = "mlp",
        gumbel_tau: float = 1.0,
        lora_rank: int = 16,
    ):
        super().__init__()
        self.k_steps = k_steps
        self.n_actions = n_actions
        self.gumbel_tau = gumbel_tau
        self.stop_id = ACTION2ID["STOP"]
        self.resampler = Resampler(in_dim, n_tokens, d_model)
        self.q_proj = nn.Linear(in_dim, d_model)
        self.a_proj = nn.Linear(in_dim, d_model)
        self.action_emb = nn.Embedding(n_actions, d_model)
        self.policy = nn.Sequential(
            nn.Linear(d_model * 4, d_model),
            nn.ReLU(),
            nn.Linear(d_model, n_actions),
        )
        self.reader = CrossRead(d_model)
        self.lora_q = ActionLoRA(d_model, n_actions, rank=lora_rank)
        self.lora_m = ActionLoRA(d_model, n_actions, rank=lora_rank)
        self.gru = nn.GRUCell(d_model * 2, d_model)
        self.score = nn.Linear(d_model, d_model)

    def _sample_action(self, logits: torch.Tensor, hard: bool = True):
        if self.training:
            y_soft = F.gumbel_softmax(logits, tau=self.gumbel_tau, hard=hard, dim=-1)
        else:
            ids = logits.argmax(dim=-1)
            y_soft = F.one_hot(ids, num_classes=self.n_actions).float()
        return y_soft, y_soft.argmax(dim=-1)

    def forward(
        self,
        mem_raw,
        q_emb,
        cand_emb,
        force_actions=None,
        shuffle_actions: bool = False,
    ):
        m = self.resampler(mem_raw)
        q = self.q_proj(q_emb)
        u = q
        r = torch.zeros_like(u)
        a_prev = torch.zeros(u.size(0), dtype=torch.long, device=u.device)
        margins, alphas, pi_list, act_list = [], [], [], []
        cand = self.a_proj(cand_emb)
        stopped = torch.zeros(u.size(0), dtype=torch.bool, device=u.device)

        for step in range(self.k_steps):
            logits_pi = self.policy(torch.cat([u, r, self.action_emb(a_prev), q], dim=-1))
            pi = logits_pi.softmax(dim=-1)
            if force_actions is not None:
                ids = force_actions[:, step]
                y_soft = F.one_hot(ids, num_classes=self.n_actions).float()
            else:
                y_soft, ids = self._sample_action(logits_pi, hard=True)
                if shuffle_actions:
                    # shuffle the *sequence position* actions across the batch
                    perm = torch.randperm(ids.size(0), device=ids.device)
                    ids = ids[perm]
                    y_soft = F.one_hot(ids, num_classes=self.n_actions).float()

            e = y_soft @ self.action_emb.weight
            u_act = u + self.lora_q(u, ids)
            m_act = m + self.lora_m(m, ids)
            r_new, alpha = self.reader(u_act, m_act, e)
            u_new = self.gru(torch.cat([r_new, e], dim=-1), u)

            stop_now = stopped | (ids == self.stop_id)
            u = torch.where(stop_now.unsqueeze(-1), u, u_new)
            r = torch.where(stop_now.unsqueeze(-1), r, r_new)
            a_prev = torch.where(stop_now, a_prev, ids)
            stopped = stop_now

            s = torch.einsum("bd,bcd->bc", self.score(u), cand)
            margins.append(s[:, 0] - s[:, 1:].max(dim=-1).values)
            alphas.append(alpha)
            pi_list.append(pi)
            act_list.append(ids)

        return {
            "logits": torch.einsum("bd,bcd->bc", self.score(u), cand),
            "margins": torch.stack(margins, dim=1),
            "alphas": torch.stack(alphas, dim=1),
            "pis": torch.stack(pi_list, dim=1),
            "actions": torch.stack(act_list, dim=1),
            "u": u,
        }


class CounterfactualCausalNavigator(nn.Module):
    """Answer-contrast-conditioned navigation with do-removal validation."""

    def __init__(
        self,
        in_dim: int,
        d_model: int = 256,
        n_tokens: int = 16,
        k_steps: int = 4,
        accept_temperature: float = 5.0,
    ):
        super().__init__()
        self.k_steps = k_steps
        self.accept_temperature = accept_temperature
        self.resampler = Resampler(in_dim, n_tokens, d_model)
        self.q_proj = nn.Linear(in_dim, d_model)
        self.a_proj = nn.Linear(in_dim, d_model)
        # [current state, previous read, top-2 answer contrast, previous causal gain]
        self.direction_policy = nn.Sequential(
            nn.Linear(d_model * 3 + 1, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.reader = CrossRead(d_model)
        self.update = nn.GRUCell(d_model * 2, d_model)
        self.score = nn.Linear(d_model, d_model)

    def _score(self, u: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bd,bcd->bc", self.score(u), candidates)

    @staticmethod
    def _top2_contrast(
        logits: torch.Tensor, candidates: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        top2 = logits.topk(k=2, dim=-1).indices
        b = torch.arange(logits.size(0), device=logits.device)
        contrast = candidates[b, top2[:, 0]] - candidates[b, top2[:, 1]]
        return F.normalize(contrast, dim=-1), top2

    @staticmethod
    def _predicted_margin(logits: torch.Tensor, top2: torch.Tensor) -> torch.Tensor:
        b = torch.arange(logits.size(0), device=logits.device)
        return logits[b, top2[:, 0]] - logits[b, top2[:, 1]]

    def forward(
        self,
        mem_raw: torch.Tensor,
        q_emb: torch.Tensor,
        cand_emb: torch.Tensor,
        intervention: str | None = None,
    ) -> dict:
        memory = self.resampler(mem_raw)
        q = self.q_proj(q_emb)
        candidates = self.a_proj(cand_emb)
        u = q
        previous_read = torch.zeros_like(u)
        previous_gain = torch.zeros(u.size(0), 1, device=u.device, dtype=u.dtype)
        margins, causal_gains, gates, alphas, directions = [], [], [], [], []

        for _ in range(self.k_steps):
            logits_before = self._score(u, candidates)
            contrast, top2 = self._top2_contrast(logits_before.detach(), candidates)
            z = F.normalize(
                self.direction_policy(
                    torch.cat([u, previous_read, contrast, previous_gain], dim=-1)
                ),
                dim=-1,
            )
            read, alpha = self.reader(u, memory, z)
            if intervention == "random_direction":
                z = F.normalize(torch.randn_like(z), dim=-1)
                read, alpha = self.reader(u, memory, z)
            elif intervention == "zero_read":
                read = torch.zeros_like(read)

            factual = self.update(torch.cat([read, z], dim=-1), u)
            # do(r_k=0), holding z and previous state fixed.
            removed = self.update(torch.cat([torch.zeros_like(read), z], dim=-1), u)
            factual_logits = self._score(factual, candidates)
            removed_logits = self._score(removed, candidates)
            causal_gain = self._predicted_margin(
                factual_logits, top2
            ) - self._predicted_margin(removed_logits, top2)

            # Accept a read only when it causally helps the current top hypothesis.
            gate = torch.sigmoid(self.accept_temperature * causal_gain).unsqueeze(-1)
            if intervention == "accept_all":
                gate = torch.ones_like(gate)
            elif intervention == "reject_all":
                gate = torch.zeros_like(gate)
            u = gate * factual + (1.0 - gate) * u
            previous_read = gate * read + (1.0 - gate) * previous_read
            previous_gain = causal_gain.detach().unsqueeze(-1)

            scores = self._score(u, candidates)
            margins.append(scores[:, 0] - scores[:, 1:].max(dim=-1).values)
            causal_gains.append(causal_gain)
            gates.append(gate.squeeze(-1))
            alphas.append(alpha)
            directions.append(z)

        return {
            "logits": self._score(u, candidates),
            "margins": torch.stack(margins, dim=1),
            "causal_gains": torch.stack(causal_gains, dim=1),
            "gates": torch.stack(gates, dim=1),
            "alphas": torch.stack(alphas, dim=1),
            "directions": torch.stack(directions, dim=1),
            "pis": None,
            "actions": None,
            "u": u,
        }


class OneShotReader(nn.Module):
    def __init__(self, in_dim, d_model=256, n_tokens=16):
        super().__init__()
        self.resampler = Resampler(in_dim, n_tokens, d_model)
        self.q_proj = nn.Linear(in_dim, d_model)
        self.a_proj = nn.Linear(in_dim, d_model)
        self.reader = CrossRead(d_model)
        self.score = nn.Linear(d_model, d_model)

    def forward(self, mem_raw, q_emb, cand_emb):
        m = self.resampler(mem_raw)
        u = self.q_proj(q_emb)
        r, alpha = self.reader(u, m, None)
        u = u + r
        cand = self.a_proj(cand_emb)
        logits = torch.einsum("bd,bcd->bc", self.score(u), cand)
        m0 = logits[:, 0] - logits[:, 1:].max(dim=-1).values
        return {"logits": logits, "margins": m0.unsqueeze(1), "alphas": alpha.unsqueeze(1), "pis": None, "actions": None, "u": u}


class PlainRecurrent(nn.Module):
    def __init__(self, in_dim, d_model=256, n_tokens=16, k_steps=4):
        super().__init__()
        self.k_steps = k_steps
        self.resampler = Resampler(in_dim, n_tokens, d_model)
        self.q_proj = nn.Linear(in_dim, d_model)
        self.a_proj = nn.Linear(in_dim, d_model)
        self.reader = CrossRead(d_model)
        self.gru = nn.GRUCell(d_model, d_model)
        self.score = nn.Linear(d_model, d_model)

    def forward(self, mem_raw, q_emb, cand_emb):
        m = self.resampler(mem_raw)
        u = self.q_proj(q_emb)
        margins = []
        cand = self.a_proj(cand_emb)
        for _ in range(self.k_steps):
            r, _ = self.reader(u, m, None)
            u = self.gru(r, u)
            s = torch.einsum("bd,bcd->bc", self.score(u), cand)
            margins.append(s[:, 0] - s[:, 1:].max(dim=-1).values)
        logits = torch.einsum("bd,bcd->bc", self.score(u), cand)
        return {
            "logits": logits,
            "margins": torch.stack(margins, dim=1),
            "alphas": None,
            "pis": None,
            "actions": None,
            "u": u,
        }


def progress_loss(margins: torch.Tensor, delta: float = 0.05) -> torch.Tensor:
    losses = []
    for k in range(1, margins.size(1)):
        losses.append(F.relu(margins[:, k - 1] + delta - margins[:, k]))
    return torch.stack(losses, dim=1).mean() if losses else margins.new_zeros(())


def complementarity_loss(alphas: torch.Tensor | None) -> torch.Tensor:
    if alphas is None:
        return torch.tensor(0.0)
    loss = 0.0
    count = 0
    for k in range(1, alphas.size(1)):
        loss = loss + (alphas[:, k] * alphas[:, k - 1]).sum(dim=-1).mean()
        count += 1
    return loss / max(count, 1)


def action_sketch_loss(pis: torch.Tensor | None, sketches: list[list[str]], k_steps: int) -> torch.Tensor:
    if pis is None:
        return torch.tensor(0.0)
    targets = []
    for sk in sketches:
        ids = [ACTION2ID.get(a, ACTION2ID["COMPOSE"]) for a in sk[:k_steps]]
        while len(ids) < k_steps:
            ids.append(ACTION2ID["STOP"])
        targets.append(ids[:k_steps])
    tgt = torch.tensor(targets, device=pis.device)
    b, k, a = pis.shape
    return F.cross_entropy(pis.reshape(b * k, a), tgt.reshape(b * k))


def entropy_bonus(pis: torch.Tensor | None) -> torch.Tensor:
    """Maximize entropy early so policy does not collapse to one sketch."""
    if pis is None:
        return torch.tensor(0.0)
    # pis: [B,K,A]
    ent = -(pis * (pis.clamp_min(1e-8).log())).sum(dim=-1).mean()
    return ent


def causal_necessity_loss(
    causal_gains: torch.Tensor | None, gamma: float = 0.05
) -> torch.Tensor:
    """Require each accepted read to improve the current answer contrast."""
    if causal_gains is None:
        return torch.tensor(0.0)
    return F.relu(gamma - causal_gains).mean()


def direction_novelty_loss(directions: torch.Tensor | None) -> torch.Tensor:
    """Discourage consecutive causal directions from collapsing."""
    if directions is None or directions.size(1) < 2:
        if directions is None:
            return torch.tensor(0.0)
        return directions.new_zeros(())
    similarities = F.cosine_similarity(
        directions[:, 1:], directions[:, :-1], dim=-1
    )
    return similarities.abs().mean()
