"""Torch datasets for immutable four-slot features and frozen QF1 caches."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from .contracts import SLOT_NAMES
from .feature_store import FourSlotFeatureStore
from .splits import video_disjoint_indices


class QF1FeatureDataset(Dataset[dict[str, object]]):
    def __init__(self, store: FourSlotFeatureStore, indices: Sequence[int] | None = None) -> None:
        self.store = store
        self.indices = tuple(indices if indices is not None else range(len(store)))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.store.row(self.indices[index])
        return {
            "node_id": row["node_id"],
            "video_id": row["video_id"],
            "features": {
                # Feature-store arrays are read-only memory maps.  Copy each
                # selected row before exposing it to torch so an accidental
                # in-place operation cannot mutate undefined read-only memory.
                name: torch.from_numpy(
                    np.array(row["features"][name], copy=True)
                ).float()
                for name in SLOT_NAMES
            },
            "validity": torch.from_numpy(
                np.array(row["validity"], copy=True)
            ).bool(),
        }


def collate_qf1(rows: Sequence[dict[str, object]]) -> dict[str, object]:
    if not rows:
        raise ValueError("cannot collate an empty QF1 batch")
    features = {
        name: torch.stack(
            [row["features"][name] for row in rows]  # type: ignore[index]
        )
        for name in SLOT_NAMES
    }
    return {
        "node_ids": tuple(str(row["node_id"]) for row in rows),
        "video_ids": tuple(str(row["video_id"]) for row in rows),
        "features": features,
        "validity": torch.stack([row["validity"] for row in rows]),  # type: ignore[list-item]
    }


@dataclass(frozen=True)
class RetrievalLabels:
    positive: Tensor
    trusted_negative: Tensor

    def __post_init__(self) -> None:
        if self.positive.dtype is not torch.bool or self.trusted_negative.dtype is not torch.bool:
            raise ValueError("retrieval label masks must be boolean")
        if self.positive.shape != self.trusted_negative.shape:
            raise ValueError("retrieval label masks must have equal shapes")
        if (self.positive & self.trusted_negative).any():
            raise ValueError("retrieval labels overlap")

