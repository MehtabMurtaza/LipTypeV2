from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class FaceLandmarks:
    """Face landmarks in normalized image coordinates (x,y in [0,1])."""

    xy: np.ndarray  # [N,2] float32


class LandmarksBackend(ABC):
    @abstractmethod
    def detect(self, frame_rgb: np.ndarray) -> Optional[FaceLandmarks]:
        """Return landmarks for the most prominent face or None."""


class NullLandmarksBackend(LandmarksBackend):
    """Fallback backend that disables landmarks (always returns None)."""

    def detect(self, frame_rgb: np.ndarray) -> Optional[FaceLandmarks]:
        return None


class MediaPipeFaceMesh(LandmarksBackend):
    """MediaPipe FaceMesh backend (468 landmarks)."""

    def __init__(self, static_image_mode: bool = False, max_num_faces: int = 1):
        import mediapipe as mp

        self._mp = mp
        if not hasattr(mp, "solutions"):
            raise RuntimeError(
                "This mediapipe build does not provide mp.solutions.*. "
                "Use NullLandmarksBackend or a tasks-based landmark model."
            )
        self._face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=static_image_mode,
            max_num_faces=max_num_faces,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def detect(self, frame_rgb: np.ndarray) -> Optional[FaceLandmarks]:
        res = self._face_mesh.process(frame_rgb)
        if not res.multi_face_landmarks:
            return None
        lm = res.multi_face_landmarks[0]
        xy = np.array([(p.x, p.y) for p in lm.landmark], dtype=np.float32)
        return FaceLandmarks(xy=xy)


def default_landmarks_backend() -> LandmarksBackend:
    # Prefer MediaPipe solutions API if available; otherwise fall back to center-crop ROI.
    try:
        return MediaPipeFaceMesh(static_image_mode=False, max_num_faces=1)
    except Exception:
        return NullLandmarksBackend()


class Dlib68Backend(LandmarksBackend):
    """dlib 68-point landmark backend.

    Requires a shape predictor file, e.g. `shape_predictor_68_face_landmarks.dat`.
    Returns 68 landmarks in normalized coordinates.
    """

    def __init__(self, predictor_path: str):
        import dlib

        self._dlib = dlib
        self._detector = dlib.get_frontal_face_detector()
        self._predictor = dlib.shape_predictor(predictor_path)

    def detect(self, frame_rgb: np.ndarray) -> Optional[FaceLandmarks]:
        # dlib expects uint8 ndarray; it can handle RGB.
        dets = self._detector(frame_rgb, 1)
        if not dets:
            return None
        # pick the largest face
        det = max(dets, key=lambda r: r.width() * r.height())
        shape = self._predictor(frame_rgb, det)
        pts = np.array([(shape.part(i).x, shape.part(i).y) for i in range(68)], dtype=np.float32)
        h, w = frame_rgb.shape[0], frame_rgb.shape[1]
        if w <= 0 or h <= 0:
            return None
        xy = np.stack([pts[:, 0] / float(w), pts[:, 1] / float(h)], axis=1).astype(np.float32)
        return FaceLandmarks(xy=xy)

