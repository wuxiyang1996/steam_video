"""Matched uniform, visual, and frozen-Q-Former clip retrieval."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Literal

import numpy as np


QUESTION_PROMPT = "Represent this question for video-memory evidence retrieval."
Arm = Literal["uniform", "visual", "qformer"]


class ThreeBenchRetriever:
    """Retrieve from one question-independent, identical clip candidate pool."""

    def __init__(
        self,
        *,
        steam_root: Path,
        feature_manifest: Path,
        projected_manifest: Path,
        clip_sidecar: Path,
        checkpoint: Path,
        embedding_model: str,
        device: str,
    ) -> None:
        import torch
        from sentence_transformers import SentenceTransformer

        steam_root = steam_root.expanduser().resolve()
        if str(steam_root) not in sys.path:
            sys.path.insert(0, str(steam_root))
        from steam_video_new.implicit_world_model.reasoning_v2.qformer.feature_store import (
            FourSlotFeatureStore,
        )
        from steam_video_new.implicit_world_model.reasoning_v2.qformer.model import (
            ConditionedProposalQFormer,
            GatedFourFeatureResidualScorer,
            NodeScoreHead,
        )
        from steam_video_new.implicit_world_model.reasoning_v2.qformer.qf1_cache import QF1Cache

        self.device = torch.device(device)
        self.features = FourSlotFeatureStore(feature_manifest)
        self.projected = QF1Cache(
            projected_manifest, feature_manifest_path=feature_manifest
        )
        self.embedder = SentenceTransformer(embedding_model, device=device)
        self.by_path: dict[str, list[dict[str, Any]]] = {}
        with clip_sidecar.expanduser().resolve().open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                self.by_path.setdefault(str(row["video_path"]), []).append(row)
        for rows in self.by_path.values():
            rows.sort(key=lambda row: (float(row["time_span"]["start_s"]), row["node_id"]))

        hidden_size = self.projected.manifest.hidden_size
        question_dimension = self.features.manifest.matrices["visual"].dimension
        self.modules = {
            "residual": GatedFourFeatureResidualScorer(
                question_dimension,
                enable_caption=False,
                enable_entity=False,
                enable_time=False,
                enable_qf=True,
            ).to(self.device),
            "qf_question": torch.nn.Sequential(
                torch.nn.LayerNorm(question_dimension), torch.nn.Linear(question_dimension, hidden_size)
            ).to(self.device),
            "qf2": ConditionedProposalQFormer(
                hidden_size=hidden_size, num_queries=8, num_layers=4, num_heads=12
            ).to(self.device),
            "qf_scorer": NodeScoreHead(hidden_size).to(self.device),
        }
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if payload.get("variant") != "a5_visual_four_token_qf2":
            raise ValueError("checkpoint is not the frozen A5 Q-Former residual variant")
        for name, module in self.modules.items():
            module.load_state_dict(payload["modules"][name], strict=True)
            module.eval()

    def encode_question(self, question_text: str) -> np.ndarray:
        value = self.embedder.encode(
            [question_text],
            prompt=QUESTION_PROMPT,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(value[0], dtype=np.float32)

    def retrieve(
        self,
        *,
        video_path: str,
        question_text: str,
        visible_until_s: float | None,
        arm: Arm,
        top_k: int,
    ) -> list[dict[str, Any]]:
        candidates = [
            row
            for row in self.by_path.get(str(video_path), ())
            if visible_until_s is None
            or float(row["time_span"]["start_s"]) < float(visible_until_s)
        ]
        if not candidates:
            raise ValueError(f"no visible candidate clips for {video_path}")
        count = min(top_k, len(candidates))
        if arm == "uniform":
            indices = np.linspace(0, len(candidates) - 1, count).round().astype(int)
            selected = [(candidates[int(index)], 0.0) for index in indices]
        else:
            question = self.encode_question(question_text)
            scores = self._visual_scores(candidates, question)
            if arm == "qformer":
                scores = self._qformer_scores(candidates, question, scores)
            order = np.argsort(-scores, kind="stable")[:count]
            selected = [(candidates[int(index)], float(scores[int(index)])) for index in order]
        output = []
        for rank, (row, score) in enumerate(selected, start=1):
            start = float(row["time_span"]["start_s"])
            end = float(row["time_span"]["end_s"])
            if visible_until_s is not None:
                end = min(end, float(visible_until_s))
            output.append(
                {
                    "rank": rank,
                    "score": score,
                    "clip": {
                        "clip_id": row["clip_id"],
                        "node_id": row["node_id"],
                        "video_id": row["source_video_id"],
                        "video_path": row["video_path"],
                        "start_s": start,
                        "end_s": end,
                        "granularity": "fine",
                    },
                }
            )
        return output

    def _visual_scores(self, rows: list[dict[str, Any]], question: np.ndarray) -> np.ndarray:
        values = []
        for row in rows:
            feature = self.features.node(row["node_id"])
            if not bool(feature["validity"][2]):
                values.append(float("-inf"))
                continue
            # FeatureStore uses read-only mmap arrays; normalization must not
            # mutate the persisted feature artifact.
            visual = np.array(feature["features"]["visual"], dtype=np.float32, copy=True)
            visual /= max(float(np.linalg.norm(visual)), 1e-12)
            values.append(float(visual @ question))
        return np.asarray(values, dtype=np.float32)

    def _qformer_scores(
        self, rows: list[dict[str, Any]], question: np.ndarray, visual_scores: np.ndarray
    ) -> np.ndarray:
        import torch

        nodes = torch.from_numpy(
            np.stack([np.asarray(self.projected.node(row["node_id"])) for row in rows])
        ).float().to(self.device)
        question_tensor = torch.from_numpy(question).float().unsqueeze(0).to(self.device)
        semantic = torch.zeros(1, len(rows), 3, device=self.device)
        semantic[0, :, 2] = torch.from_numpy(visual_scores).to(self.device)
        time_values = torch.from_numpy(
            np.stack(
                [np.asarray(self.features.node(row["node_id"])["features"]["time"]) for row in rows]
            )
        ).float().unsqueeze(0).to(self.device)
        with torch.inference_mode():
            qf_question = self.modules["qf_question"](question_tensor).unsqueeze(1)
            proposals = self.modules["qf2"](nodes, qf_question.expand(len(rows), -1, -1))
            qf_score = self.modules["qf_scorer"](proposals.unsqueeze(0), qf_question)
            scores, _ = self.modules["residual"](
                question_tensor, semantic, time_values, qf_score=qf_score
            )
        return scores[0].cpu().numpy()
