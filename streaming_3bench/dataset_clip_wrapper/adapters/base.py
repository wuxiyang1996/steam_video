"""Dataset adapter base types."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from ..schemas import DatasetName


@dataclass
class RawDatasetItem:
  dataset: DatasetName
  example_id: str
  split: str
  task_family: str
  video_id: str
  video_path: Path | None
  duration_s: float | None
  question: dict[str, Any]
  subtitle_paths: list[Path]
  annotation_segments: list[dict[str, Any]]
  evidence_seeds: list[dict[str, Any]]
  hidden_supervision_sources: list[str]
  raw_source_refs: list[dict[str, Any]]
  metadata: dict[str, Any] = None  # type: ignore[assignment]

  def __post_init__(self) -> None:
    if self.metadata is None:
      self.metadata = {}


class DatasetAdapter(ABC):
  name: DatasetName

  def __init__(self, dataset_root: Path, split: str = "train"):
    self.dataset_root = dataset_root
    self.split = split

  @abstractmethod
  def iter_items(self, limit: int | None = None) -> Iterator[RawDatasetItem]:
    ...
