from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np


@dataclass(frozen=True)
class VideoFrames:
    frames_rgb: np.ndarray  # [T,H,W,3], uint8
    fps: float


def read_video_rgb(path: str | Path, max_frames: int | None = None) -> VideoFrames:
    """Read a video using OpenCV and return RGB uint8 frames."""
    import cv2

    p = str(path)
    cap = cv2.VideoCapture(p)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {p}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frames: list[np.ndarray] = []
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
            if max_frames is not None and len(frames) >= max_frames:
                break
    finally:
        cap.release()
    if not frames:
        raise RuntimeError(f"No frames decoded from video: {p}")
    return VideoFrames(frames_rgb=np.stack(frames, axis=0).astype(np.uint8), fps=float(fps))


def iter_video_paths(root: str | Path, pattern: str = "*.mpg") -> Iterator[Path]:
    root = Path(root)
    yield from root.rglob(pattern)

