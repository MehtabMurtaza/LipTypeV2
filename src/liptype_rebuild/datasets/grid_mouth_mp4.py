from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tqdm import tqdm

from liptype_rebuild.datasets.align import align_to_sentence, parse_align_file
from liptype_rebuild.datasets.grid import GridLayout, SplitSpec
from liptype_rebuild.datasets.labels import Charset
from liptype_rebuild.datasets.tfrecords import ExampleSpec, make_example
from liptype_rebuild.preprocess.video_io import read_video_rgb


def _writer_for(path: Path):
    import tensorflow as tf

    path.parent.mkdir(parents=True, exist_ok=True)
    return tf.io.TFRecordWriter(str(path))


@dataclass(frozen=True)
class GridMouthMp4Config:
    """How to locate aligned mouth videos produced by Auto-AVSR."""

    # Auto-AVSR crop script mirrors input_root-relative paths under output_root,
    # but replaces the extension by default (e.g. .mpg -> .mp4).
    mouth_ext: str = "mp4"


def _as_rgb_uint8(fr: np.ndarray) -> np.ndarray:
    # `read_video_rgb` should already return RGB uint8, but guard for edge cases.
    if fr.ndim == 2:
        fr = np.stack([fr, fr, fr], axis=-1)
    if fr.shape[-1] == 1:
        fr = np.repeat(fr, 3, axis=-1)
    return fr.astype(np.uint8, copy=False)


def _resize_frames(frames_rgb: np.ndarray, *, out_w: int, out_h: int) -> np.ndarray:
    import cv2

    resized: list[np.ndarray] = []
    for fr in frames_rgb:
        fr = _as_rgb_uint8(fr)
        fr2 = cv2.resize(fr, (int(out_w), int(out_h)), interpolation=cv2.INTER_AREA)
        resized.append(fr2.astype(np.uint8, copy=False))
    return np.stack(resized, axis=0).astype(np.uint8, copy=False)


def convert_grid_mouth_mp4_to_tfrecords(
    *,
    grid_root: Path,
    mouth_root: Path,
    output_root: Path,
    split: SplitSpec,
    cfg: GridMouthMp4Config = GridMouthMp4Config(),
    num_shards: int = 64,
    max_frames: int = 75,
    img_w: int = 100,
    img_h: int = 50,
    img_c: int = 3,
    max_text_len: int = 32,
    max_examples: int | None = None,
    progress_every: int = 250,
):
    """Create GRID TFRecords from pre-cropped mouth-only mp4s.

    Expected inputs:
    - GRID layout under `grid_root`:
        - s*_processed/*.mpg
        - s*_processed/align/*.align
    - Auto-AVSR outputs under `mouth_root`, mirroring `grid_root` paths:
        - s*_processed/*.mp4  (extension configurable via cfg.mouth_ext)

    Writes:
      - output_root/train-00000-of-XXXXX.tfrecord
      - output_root/val-00000-of-XXXXX.tfrecord
      - output_root/test-00000-of-XXXXX.tfrecord
      - output_root/meta.json
    """
    import json
    import time

    if int(img_c) != 3:
        raise ValueError("Only RGB (img_c=3) is supported for mouth mp4 inputs.")

    spec = ExampleSpec(
        max_frames=int(max_frames),
        height=int(img_h),
        width=int(img_w),
        channels=int(img_c),
        max_text_len=int(max_text_len),
    )

    layout = GridLayout(root=grid_root)
    charset = Charset()

    utterances = list(layout.iter_utterances())
    total = len(utterances)
    print(
        f"[grid_mouth_mp4] utterances={total} grid_root={grid_root} mouth_root={mouth_root} out={output_root}",
        flush=True,
    )

    writers_train = [
        _writer_for(output_root / f"train-{i:05d}-of-{num_shards:05d}.tfrecord") for i in range(int(num_shards))
    ]
    writers_val = [
        _writer_for(output_root / f"val-{i:05d}-of-{num_shards:05d}.tfrecord") for i in range(int(num_shards))
    ]
    writers_test = [
        _writer_for(output_root / f"test-{i:05d}-of-{num_shards:05d}.tfrecord") for i in range(int(num_shards))
    ]

    n_train = 0
    n_val = 0
    n_test = 0
    n_missing = 0
    n_fail = 0

    mouth_ext = str(cfg.mouth_ext).strip().lstrip(".").lower()
    t0 = time.time()
    n_seen = 0
    try:
        it = utterances
        if max_examples is not None:
            it = utterances[: int(max_examples)]

        for idx, (speaker_id, video_path, align_path) in enumerate(
            tqdm(it, desc="grid_mouth", total=len(it), mininterval=2.0)
        ):
            if max_examples is not None and idx >= int(max_examples):
                break

            split_name = split.assign_split(speaker_id, video_path.stem)
            if split_name == "skip":
                continue

            rel = video_path.relative_to(grid_root)
            mouth_path = (mouth_root / rel).with_suffix(f".{mouth_ext}")
            if not mouth_path.exists():
                n_missing += 1
                n_seen += 1
                if progress_every > 0 and (n_seen % int(progress_every) == 0):
                    dt = max(1e-6, time.time() - t0)
                    rate = n_seen / dt
                    print(
                        f"[grid_mouth_mp4] seen={n_seen}/{len(it)} ok={n_train+n_val+n_test} "
                        f"missing={n_missing} fail={n_fail} rate={rate:.2f}/s (last_missing={mouth_path})",
                        flush=True,
                    )
                continue

            try:
                items = parse_align_file(str(align_path))
                sentence = align_to_sentence(items)
                label = charset.text_to_labels(sentence)

                vf = read_video_rgb(mouth_path, max_frames=None)
                frames = vf.frames_rgb
                rois = _resize_frames(frames, out_w=int(img_w), out_h=int(img_h))

                ex = make_example(
                    frames_uint8=rois,
                    label=label,
                    utterance_id=video_path.stem,
                    speaker_id=speaker_id,
                    spec=spec,
                )

                shard = idx % int(num_shards)
                if split_name == "val":
                    writers_val[shard].write(ex.SerializeToString())
                    n_val += 1
                elif split_name == "test":
                    writers_test[shard].write(ex.SerializeToString())
                    n_test += 1
                else:
                    writers_train[shard].write(ex.SerializeToString())
                    n_train += 1
            except Exception:
                n_fail += 1
            finally:
                n_seen += 1
                if progress_every > 0 and (n_seen % int(progress_every) == 0):
                    dt = max(1e-6, time.time() - t0)
                    rate = n_seen / dt
                    print(
                        f"[grid_mouth_mp4] seen={n_seen}/{len(it)} train={n_train} val={n_val} test={n_test} "
                        f"missing={n_missing} fail={n_fail} rate={rate:.2f}/s",
                        flush=True,
                    )
    finally:
        for w in writers_train + writers_val + writers_test:
            w.close()

    meta: dict = {
        "train_examples": n_train,
        "val_examples": n_val,
        "test_examples": n_test,
        "missing_mouth_videos": n_missing,
        "failed_examples": n_fail,
        "total_utterances": total,
        "num_shards": int(num_shards),
        "spec": spec.__dict__,
        "grid_root": str(grid_root),
        "mouth_root": str(mouth_root),
        "mouth_ext": mouth_ext,
        "mode": split.mode,
        "val_speakers": sorted(list(split.val_speakers)),
        "test_speakers": sorted(list(split.test_speakers)),
        "exclude_speakers": sorted(list(split.exclude_speakers)),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta

