"""Frozen text encoders for fixed memory M and query/candidate embeddings."""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class FrozenTextEncoder(nn.Module):
    """Mean-pool (or last-token) encoder. Parameters are frozen."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        max_length: int = 256,
        pool: str = "mean",
        device: str | torch.device = "cpu",
    ):
        super().__init__()
        self.model_name = model_name
        self.max_length = max_length
        self.pool = pool
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or self.tokenizer.unk_token
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()
        self.to(device)
        self.device = torch.device(device)
        self.out_dim = int(getattr(self.model.config, "hidden_size", 768))

    def _pool(self, last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.pool == "last":
            # last non-pad token
            lengths = attention_mask.sum(dim=1) - 1
            lengths = lengths.clamp(min=0)
            idx = lengths.view(-1, 1, 1).expand(-1, 1, last_hidden.size(-1))
            return last_hidden.gather(1, idx).squeeze(1)
        mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)
        summed = (last_hidden * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1e-6)
        return summed / denom

    @torch.no_grad()
    def encode(self, texts: list[str], batch_size: int = 16) -> torch.Tensor:
        outs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            toks = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            toks = {k: v.to(self.device) for k, v in toks.items()}
            out = self.model(**toks)
            hidden = out.last_hidden_state
            pooled = self._pool(hidden, toks["attention_mask"])
            outs.append(pooled)
        return torch.cat(outs, dim=0)

    @torch.no_grad()
    def encode_tokens(
        self,
        texts: list[str],
        batch_size: int = 4,
        max_length: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return padded token hidden states and masks on CPU.

        Unlike ``encode``, this preserves entity/relation/token information for
        the query-independent Perceiver memory writer.
        """
        hidden_batches = []
        mask_batches = []
        length = max_length or self.max_length
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            tokens = self.tokenizer(
                batch,
                padding="max_length",
                truncation=True,
                max_length=length,
                return_tensors="pt",
            )
            tokens = {key: value.to(self.device) for key, value in tokens.items()}
            output = self.model(**tokens)
            hidden_batches.append(output.last_hidden_state.detach().cpu().half())
            mask_batches.append(tokens["attention_mask"].detach().cpu().bool())
        return torch.cat(hidden_batches, dim=0), torch.cat(mask_batches, dim=0)

    def encode_paragraphs_as_memory(
        self,
        context: str,
        n_tokens: int = 32,
        max_paras: int = 16,
    ) -> torch.Tensor:
        """Split context into paragraphs → encode → pad/truncate to N memory tokens."""
        paras = [p.strip() for p in context.split("\n") if p.strip()]
        if not paras:
            paras = [context[:512] or "empty"]
        paras = paras[:max_paras]
        emb = self.encode(paras, batch_size=8)  # [P, d]
        d = emb.size(-1)
        if emb.size(0) >= n_tokens:
            return emb[:n_tokens]
        pad = torch.zeros(n_tokens - emb.size(0), d, device=emb.device, dtype=emb.dtype)
        return torch.cat([emb, pad], dim=0)


ENCODER_PRESETS = {
    "minilm": {
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "pool": "mean",
        "max_length": 256,
    },
    "bert": {
        "model_name": "bert-base-uncased",
        "pool": "mean",
        "max_length": 256,
    },
    "qwen3-emb-0.6b": {
        "model_name": "Qwen/Qwen3-Embedding-0.6B",
        "pool": "last",
        "max_length": 512,
    },
    "qwen3-emb-2b": {
        # NOTE: AutoModel mapping for Qwen3-VL-Embedding-2B is currently unreliable
        # (randomly initialized LM weights). Prefer qwen3-emb-0.6b for text proxy until
        # the official Qwen3VLEmbedder API is wired in.
        "model_name": "Qwen/Qwen3-Embedding-0.6B",
        "pool": "last",
        "max_length": 512,
    },
}


def build_encoder(preset: str, device: str) -> FrozenTextEncoder:
    if preset not in ENCODER_PRESETS:
        raise ValueError(f"Unknown encoder preset {preset}; choose from {list(ENCODER_PRESETS)}")
    cfg = ENCODER_PRESETS[preset]
    return FrozenTextEncoder(device=device, **cfg)
