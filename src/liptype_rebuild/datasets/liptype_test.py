from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterator

import numpy as np
from tqdm import tqdm

from liptype_rebuild.datasets.labels import Charset
from liptype_rebuild.datasets.tfrecords import ExampleSpec, make_example
from liptype_rebuild.preprocess.landmarks import LandmarksBackend, default_landmarks_backend
from liptype_rebuild.preprocess.mouth_roi import MouthRoiConfig, crop_video_mouth_rois
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

