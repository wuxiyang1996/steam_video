"""Cluster-aware default paths for the video skills workspace."""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(os.environ.get("VIDEO_SKILLS_PROJECT_ROOT", "/mnt/is_data/xwu/video_skills"))
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = Path(
    os.environ.get("VIDEO_SKILLS_DATASET_ROOT", str(PROJECT_ROOT / "data" / "datasets"))
)
DEFAULT_OUTPUT_ROOT = Path(
    os.environ.get("VIDEO_SKILLS_OUTPUT_ROOT", str(PROJECT_ROOT / "outputs" / "atomic_skills_for_video"))
)
_LOCAL_KEYS_PY = REPO_ROOT / "keys.py"
DEFAULT_KEYS_PY = os.environ.get("VIDEO_SKILLS_KEYS_PY") or (
    str(_LOCAL_KEYS_PY) if _LOCAL_KEYS_PY.exists() else None
)


def output_path(*parts: str) -> Path:
    return DEFAULT_OUTPUT_ROOT.joinpath(*parts)
