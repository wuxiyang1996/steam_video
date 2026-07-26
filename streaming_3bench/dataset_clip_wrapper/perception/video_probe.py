"""Video duration probing utilities."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path


def probe_duration_s(video_path: str | Path) -> float:
    path = Path(video_path)
    if not path.exists():
        return 0.0

    try:
        import cv2  # type: ignore

        cap = cv2.VideoCapture(str(path))
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
            nframes = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
            cap.release()
            if fps > 0 and nframes > 0:
                return float(nframes) / float(fps)
    except Exception:
        pass

    ffprobe = shutil.which("ffprobe")
    if ffprobe:
        try:
            proc = subprocess.run(
                [
                    ffprobe,
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "json",
                    str(path),
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            payload = json.loads(proc.stdout)
            return float(payload["format"]["duration"])
        except Exception:
            pass
    return 0.0
