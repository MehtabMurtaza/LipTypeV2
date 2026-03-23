from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterator

import numpy as np
from tqdm import tqdm

from liptype_rebuild.datasets.labels import Charset
from liptype_rebuild.datasets.tfrecords import ExampleSpec, make_example
from liptype_rebuild.preprocess.video_io import read_video_rgb


_README_ROW_RE = re.compile(r"^P-id-SSpeech-ver(?P<ver>\d+)-Ph(?P<ph>\d+)\s+(?P<phrase>.+?)\s*$")
_FILE_RE = re.compile(
    r"^P(?P<pnum>\d{1,2})_?SSpeech-ver(?P<ver>\d+)-_?Ph(?P<ph>\d{1,2}).*\.mp4$",
    re.IGNORECASE,
)


def _clean_phrase(text: str) -> str:
    t = text.strip().lower()
    t = t.replace("-", " ")
    t = re.sub(r"[^a-z ]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def parse_readme_phrases(readme_path: Path) -> dict[tuple[int, int], str]:
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
class MouthVideo:
    participant: str
    video_path: Path
    ver: int
    ph: int
    phrase: str


def iter_mouth_videos(
    input_root: Path,
    phrases: dict[tuple[int, int], str],
    include_vers: set[int],
) -> Iterator[MouthVideo]:
    """Iterate mouth-only videos under input_root (mirrors original structure)."""
    all_mp4 = sorted([p for p in input_root.rglob("*.mp4") if p.is_file()])
    name_set = {p.name for p in all_mp4}
    for vp in all_mp4:
        # Dedup common "trailing 1" pattern only.
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

        # Infer participant from path segment if present; else from filename pnum.
        participant = next((part for part in vp.parts if part.lower().startswith("p") and part[1:].isdigit()), None)
        if participant is None:
            participant = f"P{int(m.group('pnum'))}"

        yield MouthVideo(participant=participant, video_path=vp, ver=ver, ph=ph, phrase=phrase)


def _writer_for(path: Path):
    import tensorflow as tf

    path.parent.mkdir(parents=True, exist_ok=True)
    return tf.io.TFRecordWriter(str(path))


def convert_mouth_mp4_to_tfrecords(
    input_root: Path,
    readme_path: Path,
    output_root: Path,
    include_vers: set[int],
    num_shards: int = 8,
    max_frames: int = 75,
    out_w: int = 100,
    out_h: int = 50,
    out_c: int = 3,
    max_text_len: int = 120,
    max_examples: int | None = None,
):
    """Convert mouth-only mp4s to TFRecords (test split only)."""
    import json
    import cv2

    phrases = parse_readme_phrases(readme_path)
    charset = Charset()

    spec = ExampleSpec(
        max_frames=max_frames,
        height=out_h,
        width=out_w,
        channels=out_c,
        max_text_len=max_text_len,
    )

    writers = [_writer_for(output_root / f"test-{i:05d}-of-{num_shards:05d}.tfrecord") for i in range(num_shards)]
    n = 0
    try:
        for idx, mv in enumerate(tqdm(iter_mouth_videos(input_root, phrases, include_vers), desc="mouth_mp4")):
            if max_examples is not None and idx >= max_examples:
                break
            vf = read_video_rgb(mv.video_path, max_frames=None)
            frames = vf.frames_rgb

            # Resize to LipType expected size
            resized: list[np.ndarray] = []
            for fr in frames:
                if fr.ndim == 2:
                    fr = cv2.cvtColor(fr, cv2.COLOR_GRAY2RGB)
                if fr.shape[2] == 1:
                    fr = np.repeat(fr, 3, axis=2)
                fr2 = cv2.resize(fr, (out_w, out_h), interpolation=cv2.INTER_AREA)
                resized.append(fr2.astype(np.uint8))
            rois = np.stack(resized, axis=0).astype(np.uint8)

            label = charset.text_to_labels(mv.phrase)
            ex = make_example(
                frames_uint8=rois,
                label=label,
                utterance_id=mv.video_path.stem,
                speaker_id=mv.participant,
                spec=spec,
            )
            shard = idx % num_shards
            writers[shard].write(ex.SerializeToString())
            n += 1
    finally:
        for w in writers:
            w.close()

    meta = {
        "test_examples": n,
        "num_shards": num_shards,
        "spec": spec.__dict__,
        "include_vers": sorted(list(include_vers)),
        "readme": str(readme_path),
        "input_root": str(input_root),
        "note": "Mouth-only mp4s resized to LipType input size.",
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta

