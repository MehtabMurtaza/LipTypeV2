from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
from tqdm import tqdm

from liptype_rebuild.datasets.align import align_to_sentence, parse_align_file
from liptype_rebuild.datasets.labels import Charset
from liptype_rebuild.datasets.splits import SplitConfig, make_overlapped_val_set
from liptype_rebuild.datasets.tfrecords import ExampleSpec, make_example
from liptype_rebuild.preprocess.landmarks import LandmarksBackend, default_landmarks_backend
from liptype_rebuild.preprocess.mouth_roi import MouthRoiConfig, crop_video_mouth_rois
from liptype_rebuild.preprocess.video_io import read_video_rgb


@dataclass(frozen=True)
class GridLayout:
    """Your current dataset layout: data/s*_processed/*.mpg + data/s*_processed/align/*.align"""

    root: Path

    def speaker_dirs(self) -> list[Path]:
        return sorted([p for p in self.root.iterdir() if p.is_dir() and p.name.endswith("_processed")])

    def iter_utterances(self) -> Iterator[tuple[str, Path, Path]]:
        """Yield (speaker_id, video_path, align_path)."""
        for sdir in self.speaker_dirs():
            speaker_id = sdir.name.split("_")[0]  # s1_processed -> s1
            align_dir = sdir / "align"
            if not align_dir.exists():
                continue
            for video_path in sorted(sdir.glob("*.mpg")):
                utt = video_path.stem
                align_path = align_dir / f"{utt}.align"
                if align_path.exists():
                    yield speaker_id, video_path, align_path


@dataclass(frozen=True)
class SplitSpec:
    """Split specification for converting utterances into TFRecord splits."""

    mode: str
    val_speakers: set[str]
    test_speakers: set[str]
    exclude_speakers: set[str]
    val_utterances: set[tuple[str, str]]  # (speaker_id, utterance_id) for overlapped mode

    @staticmethod
    def from_config(sc: SplitConfig, utterances: Iterable[tuple[str, str]]) -> "SplitSpec":
        if sc.mode == "overlapped_utterances":
            val_set = make_overlapped_val_set(
                utterances,
                test_speakers=sc.test_speakers,
                exclude_speakers=sc.exclude_speakers,
                val_utterances_per_speaker=sc.val_utterances_per_speaker,
                seed=sc.seed,
            )
            return SplitSpec(
                mode=sc.mode,
                val_speakers=set(sc.val_speakers),
                test_speakers=set(sc.test_speakers),
                exclude_speakers=set(sc.exclude_speakers),
                val_utterances=val_set,
            )

        # speaker_holdout (legacy)
        return SplitSpec(
            mode=sc.mode,
            val_speakers=set(sc.val_speakers),
            test_speakers=set(sc.test_speakers),
            exclude_speakers=set(sc.exclude_speakers),
            val_utterances=set(),
        )

    def assign_split(self, speaker_id: str, utterance_id: str) -> str:
        """Return one of: 'train' | 'val' | 'test' | 'skip'."""
        if speaker_id in self.exclude_speakers:
            return "skip"
        if speaker_id in self.test_speakers:
            return "test"
        if self.mode == "overlapped_utterances":
            return "val" if (speaker_id, utterance_id) in self.val_utterances else "train"
        # speaker_holdout
        return "val" if speaker_id in self.val_speakers else "train"


def _writer_for(path: Path):
    import tensorflow as tf

    path.parent.mkdir(parents=True, exist_ok=True)
    return tf.io.TFRecordWriter(str(path))


def convert_grid_to_tfrecords(
    input_root: Path,
    output_root: Path,
    split: SplitSpec,
    num_shards: int = 64,
    max_frames: int = 75,
    max_examples: int | None = None,
    roi_cfg: MouthRoiConfig = MouthRoiConfig(),
    spec: ExampleSpec | None = None,
    landmarks_backend: LandmarksBackend | None = None,
):
    """Convert dataset into sharded TFRecords.

    Writes:
      - output_root/train-00000-of-XXXXX.tfrecord
      - output_root/val-00000-of-XXXXX.tfrecord
      - output_root/test-00000-of-XXXXX.tfrecord (if any test speakers are configured; files are still created)
    """
    import json
    import tensorflow as tf

    if spec is None:
        spec = ExampleSpec(max_frames=max_frames, height=roi_cfg.height, width=roi_cfg.width, channels=3)
    if landmarks_backend is None:
        landmarks_backend = default_landmarks_backend()

    layout = GridLayout(root=input_root)
    charset = Charset()

    writers_train = [
        _writer_for(output_root / f"train-{i:05d}-of-{num_shards:05d}.tfrecord") for i in range(num_shards)
    ]
    writers_val = [
        _writer_for(output_root / f"val-{i:05d}-of-{num_shards:05d}.tfrecord") for i in range(num_shards)
    ]
    writers_test = [
        _writer_for(output_root / f"test-{i:05d}-of-{num_shards:05d}.tfrecord") for i in range(num_shards)
    ]
    n_train = 0
    n_val = 0
    n_test = 0
    try:
        for idx, (speaker_id, video_path, align_path) in enumerate(tqdm(layout.iter_utterances())):
            if max_examples is not None and idx >= max_examples:
                break
            split_name = split.assign_split(speaker_id, video_path.stem)
            if split_name == "skip":
                continue
            items = parse_align_file(str(align_path))
            sentence = align_to_sentence(items)
            label = charset.text_to_labels(sentence)

            video = read_video_rgb(video_path, max_frames=None)
            # detect landmarks per frame
            lms = [landmarks_backend.detect(frame) for frame in video.frames_rgb]
            rois = crop_video_mouth_rois(video.frames_rgb, lms, cfg=roi_cfg)  # [T,50,100,3]

            ex = make_example(
                frames_uint8=rois,
                label=label,
                utterance_id=video_path.stem,
                speaker_id=speaker_id,
                spec=spec,
            )
            shard = idx % num_shards
            if split_name == "val":
                writers_val[shard].write(ex.SerializeToString())
                n_val += 1
            elif split_name == "test":
                writers_test[shard].write(ex.SerializeToString())
                n_test += 1
            else:
                # default to train
                writers_train[shard].write(ex.SerializeToString())
                n_train += 1
    finally:
        for w in writers_train + writers_val + writers_test:
            w.close()

    meta: dict = {
        "train_examples": n_train,
        "val_examples": n_val,
        "test_examples": n_test,
        "num_shards": num_shards,
        "spec": spec.__dict__,
        "roi_cfg": roi_cfg.__dict__,
        "mode": split.mode,
        "val_speakers": sorted(list(split.val_speakers)),
        "test_speakers": sorted(list(split.test_speakers)),
        "exclude_speakers": sorted(list(split.exclude_speakers)),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta

