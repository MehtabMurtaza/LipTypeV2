from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from liptype_rebuild.preprocess.landmarks import FaceLandmarks


# MediaPipe FaceMesh lip landmark indices (outer+inner lips).
# See: https://developers.google.com/mediapipe/solutions/vision/face_landmarker
MOUTH_LANDMARKS = [
    # outer
    61,
    146,
    91,
    181,
    84,
    17,
    314,
    405,
    321,
    375,
    291,
    308,
    # inner
    78,
    95,
    88,
    178,
    87,
    14,
    317,
    402,
    318,
    324,
]

# dlib 68 landmark indices for lips are 48..67 inclusive
DLIB_MOUTH_LANDMARKS = list(range(48, 68))


@dataclass(frozen=True)
class MouthRoiConfig:
    width: int = 100
    height: int = 50
    pad_ratio_x: float = 0.20
    pad_ratio_y: float = 0.30


def _clip_int(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def crop_mouth_roi(
    frame_rgb: np.ndarray,
    landmarks: FaceLandmarks,
    cfg: MouthRoiConfig = MouthRoiConfig(),
) -> np.ndarray:
    """Crop and resize mouth ROI to (cfg.height, cfg.width)."""
    import cv2

    h, w, _ = frame_rgb.shape
    if landmarks.xy.shape[0] >= 68:
        idxs = DLIB_MOUTH_LANDMARKS
    else:
        idxs = MOUTH_LANDMARKS
    pts = landmarks.xy[idxs]  # [K,2] normalized
    xs = pts[:, 0] * w
    ys = pts[:, 1] * h
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()

    # pad bbox
    bw = x1 - x0
    bh = y1 - y0
    x0 -= cfg.pad_ratio_x * bw
    x1 += cfg.pad_ratio_x * bw
    y0 -= cfg.pad_ratio_y * bh
    y1 += cfg.pad_ratio_y * bh

    ix0 = _clip_int(int(np.floor(x0)), 0, w - 1)
    ix1 = _clip_int(int(np.ceil(x1)), 0, w - 1)
    iy0 = _clip_int(int(np.floor(y0)), 0, h - 1)
    iy1 = _clip_int(int(np.ceil(y1)), 0, h - 1)

    if ix1 <= ix0 or iy1 <= iy0:
        # fallback: center crop around lower-middle face area
        cx, cy = w // 2, int(h * 0.65)
        half_w, half_h = cfg.width // 2, cfg.height // 2
        ix0 = _clip_int(cx - half_w, 0, w - 1)
        ix1 = _clip_int(cx + half_w, 0, w - 1)
        iy0 = _clip_int(cy - half_h, 0, h - 1)
        iy1 = _clip_int(cy + half_h, 0, h - 1)

    roi = frame_rgb[iy0:iy1, ix0:ix1]
    if roi.size == 0:
        roi = np.zeros((cfg.height, cfg.width, 3), dtype=np.uint8)
    roi = cv2.resize(roi, (cfg.width, cfg.height), interpolation=cv2.INTER_AREA)
    return roi.astype(np.uint8)


def crop_video_mouth_rois(
    frames_rgb: np.ndarray,
    landmarks_seq: Iterable[FaceLandmarks | None],
    cfg: MouthRoiConfig = MouthRoiConfig(),
) -> np.ndarray:
    """Crop mouth ROIs for a whole video. Returns uint8 [T,H,W,3]."""
    import cv2

    rois: list[np.ndarray] = []
    for frame, lm in zip(frames_rgb, landmarks_seq):
        if lm is None:
            # fallback: center crop
            h, w, _ = frame.shape
            cx, cy = w // 2, int(h * 0.65)
            half_w, half_h = cfg.width // 2, cfg.height // 2
            x0 = _clip_int(cx - half_w, 0, w - 1)
            x1 = _clip_int(cx + half_w, 0, w - 1)
            y0 = _clip_int(cy - half_h, 0, h - 1)
            y1 = _clip_int(cy + half_h, 0, h - 1)
            roi = frame[y0:y1, x0:x1]
            roi = cv2.resize(roi, (cfg.width, cfg.height), interpolation=cv2.INTER_AREA)
            rois.append(roi.astype(np.uint8))
        else:
            rois.append(crop_mouth_roi(frame, lm, cfg))
    return np.stack(rois, axis=0).astype(np.uint8)

