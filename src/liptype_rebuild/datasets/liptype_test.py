from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
import re
from typing import Iterator

import numpy as np
from tqdm import tqdm

from liptype_rebuild.datasets.labels import Charset
from liptype_rebuild.datasets.tfrecords import ExampleSpec, make_example
from liptype_rebuild.preprocess.landmarks import LandmarksBackend, default_landmarks_backend
from liptype_rebuild.preprocess.mouth_roi import MouthRoiConfig, crop_video_mouth_rois
from liptype_rebuild.preprocess.paper_alignment import Ibug68WithDlibDetector, PaperPreprocConfig, process_video_paper_style
from liptype_rebuild.preprocess.video_io import read_video_rgb


_README_ROW_RE = re.compile(r"^P-id-SSpeech-ver(?P<ver>\d+)-Ph(?P<ph>\d+)\s+(?P<phrase>.+?)\s*$")

# Support both observed filename formats:
# - P01SSpeech-ver10-Ph10202023-052057.mp4
# - P02_SSpeech-ver1-_Ph18_NL202021-022052.mp4
_FILE_RE = re.compile(
    r"^P(?P<pnum>\d{1,2})_?SSpeech-ver(?P<ver>\d+)-_?Ph(?P<ph>\d{1,2}).*\.mp4$",
    re.IGNORECASE,
)


def _clean_phrase(text: str) -> str:
    # Keep only a-z and spaces (Charset drops others anyway), but do a bit of cleanup:
    t = text.strip().lower()
    t = t.replace("-", " ")
    t = re.sub(r"[^a-z ]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def parse_readme_phrases(readme_path: Path) -> dict[tuple[int, int], str]:
    """Parse README.txt mapping (ver, ph) -> phrase text."""
    mapping: dict[tuple[int, int], str] = {}
    for line in readme_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = _README_ROW_RE.match(line.strip())
        if not m:
            continue
        ver = int(m.group("ver"))
        ph = int(m.group("ph"))
        phrase = _clean_phrase(m.group("phrase"))
        mapping[(ver, ph)] = phrase
    return mapping


@dataclass(frozen=True)
class TestVideo:
    participant: str
    video_path: Path
    ver: int
    ph: int
    phrase: str


def iter_silent_speech_videos(
    input_root: Path,
    phrases: dict[tuple[int, int], str],
    include_vers: set[int],
) -> Iterator[TestVideo]:
    """Iterate raw participant videos under LipType_Test_Dataset/*/data/silent speech/*.mp4."""
    for pdir in sorted([p for p in input_root.iterdir() if p.is_dir() and p.name.lower().startswith("p")]):
        silent_dir = pdir / "data" / "silent speech"
        if not silent_dir.exists():
            continue
        all_mp4 = sorted([p for p in silent_dir.glob("*.mp4") if p.is_file()])
        name_set = {p.name for p in all_mp4}

        for vp in all_mp4:
            # Light de-dup only for the very common "same file + trailing 1" pattern.
            if vp.name.endswith("1.mp4"):
                base = vp.name[:-5] + ".mp4"
                if base in name_set:
                    continue

            m = _FILE_RE.match(vp.name)
            if not m:
                continue
            ver = int(m.group("ver"))
            if ver not in include_vers:
                continue
            ph = int(m.group("ph"))
            phrase = phrases.get((ver, ph))
            if not phrase:
                continue
            yield TestVideo(participant=pdir.name, video_path=vp, ver=ver, ph=ph, phrase=phrase)


def _writer_for(path: Path):
    import tensorflow as tf

    path.parent.mkdir(parents=True, exist_ok=True)
    return tf.io.TFRecordWriter(str(path))


def convert_liptype_test_to_tfrecords(
    input_root: Path,
    readme_path: Path,
    output_root: Path,
    include_vers: set[int],
    num_shards: int = 8,
    max_frames: int = 75,
    max_text_len: int = 120,
    roi_cfg: MouthRoiConfig = MouthRoiConfig(),
    spec: ExampleSpec | None = None,
    landmarks_backend: LandmarksBackend | None = None,
    max_examples: int | None = None,
):
    """Convert LipType participant test dataset to TFRecords (writes to test-*.tfrecord only)."""
    import json

    if spec is None:
        spec = ExampleSpec(
            max_frames=max_frames,
            height=roi_cfg.height,
            width=roi_cfg.width,
            channels=3,
            max_text_len=max_text_len,
        )
    if landmarks_backend is None:
        landmarks_backend = default_landmarks_backend()

    phrases = parse_readme_phrases(readme_path)
    charset = Charset()

    writers_test = [
        _writer_for(output_root / f"test-{i:05d}-of-{num_shards:05d}.tfrecord") for i in range(num_shards)
    ]
    n_test = 0
    try:
        for idx, tv in enumerate(
            tqdm(iter_silent_speech_videos(input_root, phrases, include_vers), desc="liptype_test")
        ):
            if max_examples is not None and idx >= max_examples:
                break
            video = read_video_rgb(tv.video_path, max_frames=None)
            frames_rgb = video.frames_rgb

            lms = [landmarks_backend.detect(frame) for frame in frames_rgb]
            rois = crop_video_mouth_rois(frames_rgb, lms, cfg=roi_cfg)

            label = charset.text_to_labels(tv.phrase)
            ex = make_example(
                frames_uint8=rois,
                label=label,
                utterance_id=tv.video_path.stem,
                speaker_id=tv.participant,
                spec=spec,
            )
            shard = idx % num_shards
            writers_test[shard].write(ex.SerializeToString())
            n_test += 1
    finally:
        for w in writers_test:
            w.close()

    meta: dict = {
        "test_examples": n_test,
        "num_shards": num_shards,
        "spec": spec.__dict__,
        "roi_cfg": roi_cfg.__dict__,
        "include_vers": sorted(list(include_vers)),
        "readme": str(readme_path),
        "input_root": str(input_root),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def convert_liptype_test_to_tfrecords_paper_style(
    input_root: Path,
    readme_path: Path,
    output_root: Path,
    include_vers: set[int],
    *,
    num_shards: int = 8,
    max_frames: int = 75,
    max_text_len: int = 120,
    max_examples: int | None = None,
    progress_every: int = 200,
    detector_device: str = "cpu",
    cfg: PaperPreprocConfig = PaperPreprocConfig(),
):
    """Convert LipType_Test_Dataset videos with paper-style preprocessing (test split only)."""
    import json
    import time

    phrases = parse_readme_phrases(readme_path)
    charset = Charset()
    spec = ExampleSpec(
        max_frames=int(max_frames),
        height=int(cfg.crop_h),
        width=int(cfg.crop_w),
        channels=3,
        max_text_len=int(max_text_len),
    )
    detector = Ibug68WithDlibDetector(device=detector_device)

    videos = list(iter_silent_speech_videos(input_root, phrases, include_vers))
    total_videos = len(videos)
    if max_examples is not None:
        videos = videos[: int(max_examples)]

    writers_test = [
        _writer_for(output_root / f"test-{i:05d}-of-{num_shards:05d}.tfrecord") for i in range(int(num_shards))
    ]
    n_seen = 0
    n_test = 0
    n_fail = 0
    t0 = time.time()
    try:
        for idx, tv in enumerate(tqdm(videos, desc="liptype_test_paper", total=len(videos))):
            try:
                vf = read_video_rgb(tv.video_path, max_frames=None)
                rois = process_video_paper_style(vf.frames_rgb, detector=detector, cfg=cfg)

                label = charset.text_to_labels(tv.phrase)
                ex = make_example(
                    frames_uint8=rois,
                    label=label,
                    utterance_id=tv.video_path.stem,
                    speaker_id=tv.participant,
                    spec=spec,
                )
                shard = idx % int(num_shards)
                writers_test[shard].write(ex.SerializeToString())
                n_test += 1
            except Exception:
                n_fail += 1
            finally:
                n_seen += 1
                if progress_every > 0 and (n_seen % int(progress_every) == 0):
                    dt = max(1e-6, time.time() - t0)
                    rate = n_seen / dt
                    print(
                        f"[liptype_test_paper] seen={n_seen}/{len(videos)} "
                        f"ok={n_test} fail={n_fail} rate={rate:.2f}/s",
                        flush=True,
                    )
    finally:
        for w in writers_test:
            w.close()

    meta: dict = {
        "pipeline": "paper_style_dlib_ibug68_kalman_affine",
        "test_examples": n_test,
        "failed_examples": n_fail,
        "seen_examples": n_seen,
        "total_candidates_before_limit": total_videos,
        "num_shards": int(num_shards),
        "spec": spec.__dict__,
        "paper_preproc_cfg": {
            "crop_w": cfg.crop_w,
            "crop_h": cfg.crop_h,
            "pad_ratio_x": cfg.pad_ratio_x,
            "pad_ratio_y": cfg.pad_ratio_y,
            "target_w": cfg.target_w,
            "target_h": cfg.target_h,
            "stable_points": list(cfg.stable_points),
            "canonical_68_npy": (str(cfg.canonical_68_npy) if cfg.canonical_68_npy is not None else None),
            "kalman_process_noise": cfg.kalman_process_noise,
            "kalman_measurement_noise": cfg.kalman_measurement_noise,
        },
        "include_vers": sorted(list(include_vers)),
        "readme": str(readme_path),
        "input_root": str(input_root),
        "elapsed_sec": time.time() - t0,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def convert_liptype_test_to_tfrecords_paper_style_train_val(
    input_root: Path,
    readme_path: Path,
    output_root: Path,
    include_vers: set[int],
    *,
    num_shards: int = 64,
    max_frames: int = 75,
    max_text_len: int = 120,
    val_ratio: float = 0.2,
    split_seed: int = 42,
    max_examples: int | None = None,
    progress_every: int = 200,
    detector_device: str = "cpu",
    cfg: PaperPreprocConfig = PaperPreprocConfig(),
):
    """Convert LipType_Test_Dataset videos with paper-style preprocessing (train/val only)."""
    import json
    import time

    if not (0.0 < float(val_ratio) < 1.0):
        raise ValueError(f"val_ratio must be in (0,1), got {val_ratio}")

    phrases = parse_readme_phrases(readme_path)
    charset = Charset()
    spec = ExampleSpec(
        max_frames=int(max_frames),
        height=int(cfg.crop_h),
        width=int(cfg.crop_w),
        channels=3,
        max_text_len=int(max_text_len),
    )
    detector = Ibug68WithDlibDetector(device=detector_device)

    videos = list(iter_silent_speech_videos(input_root, phrases, include_vers))
    total_videos = len(videos)
    if max_examples is not None:
        videos = videos[: int(max_examples)]

    rng = random.Random(int(split_seed))
    rng.shuffle(videos)

    n_total = len(videos)
    n_val_target = max(1, int(round(n_total * float(val_ratio)))) if n_total > 1 else 0
    n_val_target = min(n_val_target, max(0, n_total - 1))
    split_at = n_total - n_val_target
    train_videos = videos[:split_at]
    val_videos = videos[split_at:]

    writers_train = [
        _writer_for(output_root / f"train-{i:05d}-of-{num_shards:05d}.tfrecord") for i in range(int(num_shards))
    ]
    writers_val = [_writer_for(output_root / f"val-{i:05d}-of-{num_shards:05d}.tfrecord") for i in range(int(num_shards))]

    n_seen = 0
    n_train = 0
    n_val = 0
    n_fail = 0
    t0 = time.time()

    def _process_split(videos_split: list[TestVideo], split_name: str):
        nonlocal n_seen, n_train, n_val, n_fail
        for idx, tv in enumerate(tqdm(videos_split, desc=f"liptype_test_paper_{split_name}", total=len(videos_split))):
            try:
                vf = read_video_rgb(tv.video_path, max_frames=None)
                rois = process_video_paper_style(vf.frames_rgb, detector=detector, cfg=cfg)
                label = charset.text_to_labels(tv.phrase)
                ex = make_example(
                    frames_uint8=rois,
                    label=label,
                    utterance_id=tv.video_path.stem,
                    speaker_id=tv.participant,
                    spec=spec,
                )
                shard = idx % int(num_shards)
                if split_name == "train":
                    writers_train[shard].write(ex.SerializeToString())
                    n_train += 1
                else:
                    writers_val[shard].write(ex.SerializeToString())
                    n_val += 1
            except Exception:
                n_fail += 1
            finally:
                n_seen += 1
                if progress_every > 0 and (n_seen % int(progress_every) == 0):
                    dt = max(1e-6, time.time() - t0)
                    rate = n_seen / dt
                    print(
                        f"[liptype_test_paper_train_val] seen={n_seen}/{n_total} "
                        f"train={n_train} val={n_val} fail={n_fail} rate={rate:.2f}/s",
                        flush=True,
                    )

    try:
        _process_split(train_videos, "train")
        _process_split(val_videos, "val")
    finally:
        for w in writers_train + writers_val:
            w.close()

    meta: dict = {
        "pipeline": "paper_style_dlib_ibug68_kalman_affine",
        "train_examples": n_train,
        "val_examples": n_val,
        "test_examples": 0,
        "failed_examples": n_fail,
        "seen_examples": n_seen,
        "total_candidates_before_limit": total_videos,
        "split": {
            "strategy": "global_random",
            "val_ratio": float(val_ratio),
            "split_seed": int(split_seed),
            "n_total_after_limit": n_total,
        },
        "num_shards": int(num_shards),
        "spec": spec.__dict__,
        "paper_preproc_cfg": {
            "crop_w": cfg.crop_w,
            "crop_h": cfg.crop_h,
            "pad_ratio_x": cfg.pad_ratio_x,
            "pad_ratio_y": cfg.pad_ratio_y,
            "target_w": cfg.target_w,
            "target_h": cfg.target_h,
            "stable_points": list(cfg.stable_points),
            "canonical_68_npy": (str(cfg.canonical_68_npy) if cfg.canonical_68_npy is not None else None),
            "kalman_process_noise": cfg.kalman_process_noise,
            "kalman_measurement_noise": cfg.kalman_measurement_noise,
        },
        "include_vers": sorted(list(include_vers)),
        "readme": str(readme_path),
        "input_root": str(input_root),
        "elapsed_sec": time.time() - t0,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta

