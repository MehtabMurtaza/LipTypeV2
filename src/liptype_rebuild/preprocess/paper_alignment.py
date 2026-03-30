from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from liptype_rebuild.preprocess.landmarks import FaceLandmarks


@dataclass(frozen=True)
class PaperPreprocConfig:
    """Paper-style preprocessing settings.

    Output crop is fixed to 100x50 by default to mirror LipType paper text.
    """

    crop_w: int = 100
    crop_h: int = 50
    pad_ratio_x: float = 0.19
    pad_ratio_y: float = 0.30

    # Affine normalization canvas
    target_w: int = 256
    target_h: int = 256

    # Stable landmarks for affine estimation (0-based 68-point indexing)
    stable_points: tuple[int, ...] = (28, 33, 36, 39, 42, 45, 48, 54)

    # Optional canonical reference with shape [68,2] in normalized coordinates [0,1].
    # If omitted, the first valid smoothed frame is used as a fallback reference.
    canonical_68_npy: Path | None = None

    # Kalman filter noise terms (normalized coordinate system)
    kalman_process_noise: float = 1e-4
    kalman_measurement_noise: float = 5e-3


class Ibug68WithDlibDetector:
    """dlib face detector + iBug landmark predictor wrapper.

    Detection:
      - dlib frontal detector proposes face bboxes
    Landmarking:
      - ibug FAN predictor estimates facial landmarks within bboxes
    Output:
      - normalized [68,2] points (x,y in [0,1]) for the largest face
    """

    def __init__(self, device: str = "cpu", fan_model=None):
        import dlib
        from ibug.face_alignment import FANPredictor

        self._dlib = dlib
        self._detector = dlib.get_frontal_face_detector()
        # Match Auto-AVSR usage; model=None picks package default.
        self._fan = FANPredictor(device=device, model=fan_model)

    def detect(self, frame_rgb: np.ndarray) -> FaceLandmarks | None:
        h, w = frame_rgb.shape[:2]
        if h <= 0 or w <= 0:
            return None

        dets = self._detector(frame_rgb, 1)
        if not dets:
            return None

        # FAN predictor accepts an Nx5 bbox array (x1,y1,x2,y2,score).
        faces = []
        for r in dets:
            faces.append([float(r.left()), float(r.top()), float(r.right()), float(r.bottom()), 1.0])
        faces_np = np.asarray(faces, dtype=np.float32)

        try:
            points, _ = self._fan(frame_rgb, faces_np, rgb=True)
        except TypeError:
            # Older ibug versions may not expose rgb kwarg.
            points, _ = self._fan(frame_rgb, faces_np)

        if points is None or len(points) == 0:
            return None

        # Choose largest dlib bbox for consistency with previous behavior.
        areas = [(max(0.0, f[2] - f[0]) * max(0.0, f[3] - f[1])) for f in faces]
        best = int(np.argmax(np.asarray(areas))) if areas else 0
        lm = np.asarray(points[best], dtype=np.float32)
        if lm.ndim != 2 or lm.shape[1] != 2:
            return None

        # Keep first 68 points if detector returns denser landmarks.
        if lm.shape[0] < 68:
            return None
        lm = lm[:68]
        lm[:, 0] /= float(w)
        lm[:, 1] /= float(h)
        lm = np.clip(lm, 0.0, 1.0)
        return FaceLandmarks(xy=lm.astype(np.float32))


def interpolate_landmarks_seq(landmarks_seq: Iterable[FaceLandmarks | None]) -> list[FaceLandmarks] | None:
    """Fill missing landmarks via linear interpolation + edge replication."""
    seq = list(landmarks_seq)
    if not seq:
        return []

    valid = [i for i, lm in enumerate(seq) if lm is not None]
    if not valid:
        return None

    arr = [None if lm is None else lm.xy.astype(np.float32) for lm in seq]

    # Fill internal gaps linearly.
    for j in range(1, len(valid)):
        a = valid[j - 1]
        b = valid[j]
        if b - a <= 1:
            continue
        pa = arr[a]
        pb = arr[b]
        if pa is None or pb is None:
            continue
        for t in range(a + 1, b):
            alpha = (t - a) / float(b - a)
            arr[t] = (1.0 - alpha) * pa + alpha * pb

    # Edge replication.
    first = valid[0]
    last = valid[-1]
    for i in range(0, first):
        arr[i] = arr[first].copy()
    for i in range(last + 1, len(arr)):
        arr[i] = arr[last].copy()

    out = []
    for a in arr:
        if a is None:
            # Should be rare after interpolation/replication; fallback to nearest valid.
            a = arr[first].copy()
        out.append(FaceLandmarks(xy=np.clip(a, 0.0, 1.0).astype(np.float32)))
    return out


def _kalman_smooth_1d(zs: np.ndarray, q: float, r: float) -> np.ndarray:
    """Constant-velocity 1D Kalman smoother.

    State: [position, velocity]
    Measurement: position
    """
    n = int(zs.shape[0])
    if n == 0:
        return zs

    # Matrices
    A = np.array([[1.0, 1.0], [0.0, 1.0]], dtype=np.float32)
    H = np.array([[1.0, 0.0]], dtype=np.float32)
    Q = np.array([[q, 0.0], [0.0, q]], dtype=np.float32)
    R = np.array([[r]], dtype=np.float32)
    I = np.eye(2, dtype=np.float32)

    x = np.array([[zs[0]], [0.0]], dtype=np.float32)
    P = np.eye(2, dtype=np.float32) * 0.1

    out = np.zeros((n,), dtype=np.float32)
    for i in range(n):
        # Predict
        x = A @ x
        P = A @ P @ A.T + Q

        # Update
        z = np.array([[zs[i]]], dtype=np.float32)
        y = z - H @ x
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
        x = x + K @ y
        P = (I - K @ H) @ P
        out[i] = float(x[0, 0])
    return out


def kalman_smooth_landmarks_seq(
    landmarks_seq: Iterable[FaceLandmarks],
    *,
    process_noise: float,
    measurement_noise: float,
) -> list[FaceLandmarks]:
    """Apply Kalman smoothing independently on x/y per landmark index."""
    seq = list(landmarks_seq)
    if not seq:
        return []

    arr = np.stack([lm.xy for lm in seq], axis=0).astype(np.float32)  # [T,68,2]
    t, n, _ = arr.shape

    sm = np.zeros_like(arr, dtype=np.float32)
    for k in range(n):
        xs = arr[:, k, 0]
        ys = arr[:, k, 1]
        sm[:, k, 0] = _kalman_smooth_1d(xs, process_noise, measurement_noise)
        sm[:, k, 1] = _kalman_smooth_1d(ys, process_noise, measurement_noise)

    sm = np.clip(sm, 0.0, 1.0)
    return [FaceLandmarks(xy=sm[i]) for i in range(t)]


def _as_pixel_xy(lm: FaceLandmarks, w: int, h: int) -> np.ndarray:
    pts = lm.xy.astype(np.float32).copy()
    pts[:, 0] *= float(w)
    pts[:, 1] *= float(h)
    return pts


def _to_norm_xy(pts: np.ndarray, w: int, h: int) -> FaceLandmarks:
    out = pts.astype(np.float32).copy()
    out[:, 0] /= float(max(1, w))
    out[:, 1] /= float(max(1, h))
    out = np.clip(out, 0.0, 1.0)
    return FaceLandmarks(xy=out)


def _load_or_build_reference(landmarks_seq: list[FaceLandmarks], cfg: PaperPreprocConfig) -> np.ndarray:
    """Return canonical reference in pixel coordinates for affine target canvas."""
    if cfg.canonical_68_npy is not None:
        ref = np.load(str(cfg.canonical_68_npy)).astype(np.float32)
        if ref.shape != (68, 2):
            raise ValueError(f"canonical_68_npy must have shape [68,2], got {ref.shape}")
        out = ref.copy()
    else:
        # Fallback: use first frame as reference (still enables temporal stabilization).
        out = landmarks_seq[0].xy.astype(np.float32).copy()

    out[:, 0] *= float(cfg.target_w)
    out[:, 1] *= float(cfg.target_h)
    return out


def estimate_affine_to_template(
    landmarks_px: np.ndarray,
    reference_px: np.ndarray,
    stable_points: tuple[int, ...],
) -> np.ndarray:
    import cv2

    src = landmarks_px[list(stable_points)].astype(np.float32)
    dst = reference_px[list(stable_points)].astype(np.float32)
    m, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
    if m is None:
        raise RuntimeError("Affine estimation failed")
    return m.astype(np.float32)


def warp_frame_and_landmarks(
    frame_rgb: np.ndarray,
    landmarks_px: np.ndarray,
    affine_m: np.ndarray,
    target_w: int,
    target_h: int,
) -> tuple[np.ndarray, np.ndarray]:
    import cv2

    warped = cv2.warpAffine(
        frame_rgb,
        affine_m,
        dsize=(int(target_w), int(target_h)),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    # [N,2] @ [2,2]^T + [2]
    lm_warp = landmarks_px @ affine_m[:, :2].T + affine_m[:, 2]
    return warped.astype(np.uint8), lm_warp.astype(np.float32)


def extract_mouth_from_aligned(
    aligned_rgb: np.ndarray,
    aligned_landmarks_px: np.ndarray,
    cfg: PaperPreprocConfig,
) -> np.ndarray:
    import cv2

    h, w = aligned_rgb.shape[:2]
    pts = aligned_landmarks_px[48:68]  # mouth points for 68-landmark format
    xs = pts[:, 0]
    ys = pts[:, 1]

    x0, x1 = float(xs.min()), float(xs.max())
    y0, y1 = float(ys.min()), float(ys.max())
    bw = max(1.0, x1 - x0)
    bh = max(1.0, y1 - y0)
    x0 -= cfg.pad_ratio_x * bw
    x1 += cfg.pad_ratio_x * bw
    y0 -= cfg.pad_ratio_y * bh
    y1 += cfg.pad_ratio_y * bh

    ix0 = int(np.floor(np.clip(x0, 0, w - 1)))
    ix1 = int(np.ceil(np.clip(x1, 0, w - 1)))
    iy0 = int(np.floor(np.clip(y0, 0, h - 1)))
    iy1 = int(np.ceil(np.clip(y1, 0, h - 1)))

    if ix1 <= ix0 or iy1 <= iy0:
        cx, cy = w // 2, int(h * 0.65)
        hw, hh = cfg.crop_w // 2, cfg.crop_h // 2
        ix0 = max(0, cx - hw)
        ix1 = min(w, cx + hw)
        iy0 = max(0, cy - hh)
        iy1 = min(h, cy + hh)

    crop = aligned_rgb[iy0:iy1, ix0:ix1]
    if crop.size == 0:
        crop = np.zeros((cfg.crop_h, cfg.crop_w, 3), dtype=np.uint8)
    crop = cv2.resize(crop, (cfg.crop_w, cfg.crop_h), interpolation=cv2.INTER_AREA)
    return crop.astype(np.uint8)


def process_video_paper_style(
    frames_rgb: np.ndarray,
    detector: Ibug68WithDlibDetector,
    cfg: PaperPreprocConfig = PaperPreprocConfig(),
) -> np.ndarray:
    """Run paper-style landmark smoothing + affine normalization + mouth crop.

    Returns uint8 array [T, crop_h, crop_w, 3].
    """
    if frames_rgb.ndim != 4:
        raise ValueError(f"frames_rgb must be [T,H,W,3], got shape={frames_rgb.shape}")

    # 1) Detect landmarks
    detected = [detector.detect(frame) for frame in frames_rgb]

    # 2) Interpolate missing landmarks
    interp = interpolate_landmarks_seq(detected)
    if interp is None:
        # No face at all in video; return center-crop-like zeros to keep pipeline robust.
        t = int(frames_rgb.shape[0])
        return np.zeros((t, cfg.crop_h, cfg.crop_w, 3), dtype=np.uint8)

    # 3) Kalman smoothing
    smooth = kalman_smooth_landmarks_seq(
        interp,
        process_noise=float(cfg.kalman_process_noise),
        measurement_noise=float(cfg.kalman_measurement_noise),
    )

    # 4) Affine normalization + mouth crop
    reference_px = _load_or_build_reference(smooth, cfg)
    out: list[np.ndarray] = []
    for frame, lm in zip(frames_rgb, smooth):
        h, w = frame.shape[:2]
        lm_px = _as_pixel_xy(lm, w=w, h=h)
        try:
            m = estimate_affine_to_template(lm_px, reference_px, cfg.stable_points)
            aligned_rgb, aligned_lm_px = warp_frame_and_landmarks(
                frame_rgb=frame,
                landmarks_px=lm_px,
                affine_m=m,
                target_w=cfg.target_w,
                target_h=cfg.target_h,
            )
            mouth = extract_mouth_from_aligned(aligned_rgb, aligned_lm_px, cfg)
        except Exception:
            # Conservative fallback to avoid killing long conversion jobs.
            mouth = np.zeros((cfg.crop_h, cfg.crop_w, 3), dtype=np.uint8)
        out.append(mouth)
    return np.stack(out, axis=0).astype(np.uint8)

