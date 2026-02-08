"""
Parallel GRID-style preprocessing -> sharded TFRecords.

This is intentionally a *separate* entrypoint so you don't have to modify the
existing `liptype2 preprocess grid-to-tfrecords` implementation.

Parallelization strategy:
  - Enumerate all utterances once in the main process to assign a stable index.
  - Assign each utterance to a shard via: shard = idx % num_shards (same as current code).
  - Process shard-by-shard in separate worker processes.
    Each worker writes ONLY its own shard files (train-XXXXX-of-NNNNN.tfrecord and
    val-XXXXX-of-NNNNN.tfrecord), so there is no concurrent writing to the same file.

Notes:
  - This script decodes at most `--max-frames` frames per video (configurable via
    `--decode-max-frames`). This is usually what you intended, and is faster than
    decoding full videos then truncating later.

Example:
  python scripts/grid_to_tfrecords_parallel.py \
    --input-root data \
    --output-root rebuild_data/tfrecords_small2 \
    --split-config configs/grid_split_example.yaml \
    --num-shards 8 \
    --max-frames 75 \
    --max-examples 10000 \
    --workers 8 \
    --dlib-predictor assets/dlib/shape_predictor_68_face_landmarks.dat
"""

from __future__ import annotations

import json
import multiprocessing as mp
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import typer

# Allow running directly from the repo without requiring `pip install -e .`
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path and _SRC.exists():
    sys.path.insert(0, str(_SRC))

from liptype_rebuild.datasets.align import align_to_sentence, parse_align_file
from liptype_rebuild.datasets.grid import GridLayout, SplitSpec
from liptype_rebuild.datasets.labels import Charset
from liptype_rebuild.datasets.splits import load_split_config
from liptype_rebuild.datasets.tfrecords import ExampleSpec, make_example
from liptype_rebuild.preprocess.landmarks import Dlib68Backend, LandmarksBackend, default_landmarks_backend
from liptype_rebuild.preprocess.mouth_roi import MouthRoiConfig, crop_video_mouth_rois
from liptype_rebuild.preprocess.video_io import read_video_rgb


app = typer.Typer(add_completion=False, no_args_is_help=True)


def _writer_for(path: Path):
    import tensorflow as tf

    path.parent.mkdir(parents=True, exist_ok=True)
    return tf.io.TFRecordWriter(str(path))


def _iter_utterances_stable(input_root: Path) -> list[tuple[str, Path, Path]]:
    """Materialize utterance list so idx/shard assignment is stable across processes."""
    layout = GridLayout(root=input_root)
    return list(layout.iter_utterances())


def _build_backend(dlib_predictor: str | None) -> LandmarksBackend:
    if dlib_predictor:
        return Dlib68Backend(dlib_predictor)
    return default_landmarks_backend()


def _process_shard(
    shard_id: int,
    tasks: list[tuple[int, str, str, str, bool]],  # (idx, speaker_id, video_path, align_path, is_val)
    output_root: str,
    num_shards: int,
    max_frames: int,
    decode_max_frames: int | None,
    dlib_predictor: str | None,
    roi_cfg_dict: dict,
    spec_dict: dict,
) -> tuple[int, int]:
    """Write TFRecords for one shard, return (n_train, n_val)."""
    output_root_p = Path(output_root)
    roi_cfg = MouthRoiConfig(**roi_cfg_dict)
    spec = ExampleSpec(**spec_dict)
    backend = _build_backend(dlib_predictor)
    charset = Charset()

    # Create both files (even if empty) to match the existing layout.
    train_path = output_root_p / f"train-{shard_id:05d}-of-{num_shards:05d}.tfrecord"
    val_path = output_root_p / f"val-{shard_id:05d}-of-{num_shards:05d}.tfrecord"
    w_train = _writer_for(train_path)
    w_val = _writer_for(val_path)

    n_train = 0
    n_val = 0
    try:
        for _idx, speaker_id, video_path, align_path, is_val in tasks:
            items = parse_align_file(align_path)
            sentence = align_to_sentence(items)
            label = charset.text_to_labels(sentence)

            video = read_video_rgb(video_path, max_frames=decode_max_frames)
            lms = [backend.detect(frame) for frame in video.frames_rgb]
            rois = crop_video_mouth_rois(video.frames_rgb, lms, cfg=roi_cfg)  # [T,H,W,3]

            ex = make_example(
                frames_uint8=rois,
                label=label,
                utterance_id=Path(video_path).stem,
                speaker_id=speaker_id,
                spec=spec,
            )
            if is_val:
                w_val.write(ex.SerializeToString())
                n_val += 1
            else:
                w_train.write(ex.SerializeToString())
                n_train += 1
    finally:
        w_train.close()
        w_val.close()

    return n_train, n_val


@app.command("grid-to-tfrecords-parallel")
def grid_to_tfrecords_parallel(
    input_root: Path = typer.Option(..., exists=True, file_okay=False, dir_okay=True),
    output_root: Path = typer.Option(..., file_okay=False, dir_okay=True),
    split_config: Path = typer.Option(..., exists=True, dir_okay=False),
    num_shards: int = typer.Option(64, min=1),
    max_frames: int = typer.Option(75, min=1),
    max_examples: int = typer.Option(0, min=0, help="If >0, stop after N examples."),
    workers: int = typer.Option(0, min=0, help="If 0, use min(num_shards, CPU cores)."),
    decode_max_frames: int = typer.Option(
        -1,
        help="Frames to decode per video. -1 uses --max-frames; 0 decodes full video (slower).",
    ),
    dlib_predictor: Path = typer.Option(
        None, exists=True, dir_okay=False, help="Path to dlib 68-landmark predictor .dat file."
    ),
):
    """Parallel version of GRID -> TFRecords conversion (shard-per-worker)."""
    sc = load_split_config(split_config)
    split = SplitSpec.from_seen_unseen(sc.train_speakers, sc.val_speakers)

    utterances = _iter_utterances_stable(input_root)
    if max_examples and max_examples > 0:
        utterances = utterances[:max_examples]

    # Same spec/ROI defaults as the existing converter.
    roi_cfg = MouthRoiConfig()
    spec = ExampleSpec(max_frames=max_frames, height=roi_cfg.height, width=roi_cfg.width, channels=3)

    if decode_max_frames == -1:
        decode_max_frames_use: int | None = max_frames
    elif decode_max_frames == 0:
        decode_max_frames_use = None
    else:
        decode_max_frames_use = int(decode_max_frames)

    # Pre-assign tasks to shards in main process for stable distribution.
    shard_tasks: list[list[tuple[int, str, str, str, bool]]] = [[] for _ in range(num_shards)]
    for idx, (speaker_id, video_path, align_path) in enumerate(utterances):
        shard = idx % num_shards
        is_val = speaker_id in split.val_speakers
        shard_tasks[shard].append((idx, speaker_id, str(video_path), str(align_path), is_val))

    output_root.mkdir(parents=True, exist_ok=True)

    cpu_cores = mp.cpu_count() or 1
    n_workers = workers if workers and workers > 0 else min(num_shards, cpu_cores)
    n_workers = max(1, min(n_workers, num_shards))

    # Use spawn on macOS for safety with native libs (dlib/cv2/tensorflow).
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=n_workers) as pool:
        results = pool.starmap(
            _process_shard,
            [
                (
                    shard_id,
                    shard_tasks[shard_id],
                    str(output_root),
                    num_shards,
                    max_frames,
                    decode_max_frames_use,
                    (str(dlib_predictor) if dlib_predictor is not None else None),
                    asdict(roi_cfg),
                    asdict(spec),
                )
                for shard_id in range(num_shards)
            ],
        )

    n_train = sum(r[0] for r in results)
    n_val = sum(r[1] for r in results)

    meta: dict = {
        "train_examples": n_train,
        "val_examples": n_val,
        "num_shards": num_shards,
        "spec": asdict(spec),
        "roi_cfg": asdict(roi_cfg),
        "train_speakers": sorted(list(split.train_speakers)),
        "val_speakers": sorted(list(split.val_speakers)),
    }
    (output_root / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    typer.echo(f"Wrote TFRecords to {output_root} (train={n_train}, val={n_val}, shards={num_shards})")


@app.command("count-utterances")
def count_utterances(
    input_root: Path = typer.Option(..., exists=True, file_okay=False, dir_okay=True),
):
    """Print how many utterances are found under input_root."""
    utterances = _iter_utterances_stable(input_root)
    typer.echo(str(len(utterances)))


def main():
    app()


if __name__ == "__main__":
    main()

