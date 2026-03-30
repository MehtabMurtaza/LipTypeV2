from __future__ import annotations

from pathlib import Path

from tqdm import tqdm

from liptype_rebuild.datasets.align import align_to_sentence, parse_align_file
from liptype_rebuild.datasets.grid import GridLayout, SplitSpec
from liptype_rebuild.datasets.labels import Charset
from liptype_rebuild.datasets.tfrecords import ExampleSpec, make_example
from liptype_rebuild.preprocess.paper_alignment import Ibug68WithDlibDetector, PaperPreprocConfig, process_video_paper_style
from liptype_rebuild.preprocess.video_io import read_video_rgb


def _writer_for(path: Path):
    import tensorflow as tf

    path.parent.mkdir(parents=True, exist_ok=True)
    return tf.io.TFRecordWriter(str(path))


def convert_grid_to_tfrecords_paper_style(
    *,
    input_root: Path,
    output_root: Path,
    split: SplitSpec,
    num_shards: int = 64,
    max_frames: int = 75,
    max_text_len: int = 32,
    max_examples: int | None = None,
    progress_every: int = 200,
    detector_device: str = "cpu",
    cfg: PaperPreprocConfig = PaperPreprocConfig(),
):
    """Convert GRID to TFRecords using paper-style preprocessing path.

    Pipeline:
      dlib detect -> iBug 68 landmarks -> interpolation -> Kalman smoothing
      -> affine normalize -> mouth crop 100x50 -> TFRecord
    """
    import json
    import time

    layout = GridLayout(root=input_root)
    utterances = list(layout.iter_utterances())
    total_utterances = len(utterances)
    if max_examples is not None:
        utterances = utterances[: int(max_examples)]

    print(
        f"[paper_preproc] start utterances={len(utterances)} total={total_utterances} input={input_root} out={output_root}",
        flush=True,
    )
    print(
        f"[paper_preproc] crop={cfg.crop_w}x{cfg.crop_h} target={cfg.target_w}x{cfg.target_h} "
        f"kalman_q={cfg.kalman_process_noise} kalman_r={cfg.kalman_measurement_noise}",
        flush=True,
    )

    spec = ExampleSpec(
        max_frames=int(max_frames),
        height=int(cfg.crop_h),
        width=int(cfg.crop_w),
        channels=3,
        max_text_len=int(max_text_len),
    )
    charset = Charset()
    detector = Ibug68WithDlibDetector(device=detector_device)

    writers_train = [
        _writer_for(output_root / f"train-{i:05d}-of-{num_shards:05d}.tfrecord") for i in range(int(num_shards))
    ]
    writers_val = [_writer_for(output_root / f"val-{i:05d}-of-{num_shards:05d}.tfrecord") for i in range(int(num_shards))]
    writers_test = [
        _writer_for(output_root / f"test-{i:05d}-of-{num_shards:05d}.tfrecord") for i in range(int(num_shards))
    ]

    n_seen = 0
    n_train = 0
    n_val = 0
    n_test = 0
    n_fail = 0
    n_skip = 0
    t0 = time.time()

    try:
        for idx, (speaker_id, video_path, align_path) in enumerate(tqdm(utterances, desc="grid_paper", total=len(utterances))):
            split_name = split.assign_split(speaker_id, video_path.stem)
            if split_name == "skip":
                n_skip += 1
                n_seen += 1
                continue

            try:
                items = parse_align_file(str(align_path))
                sentence = align_to_sentence(items)
                label = charset.text_to_labels(sentence)

                vf = read_video_rgb(video_path, max_frames=None)
                rois = process_video_paper_style(vf.frames_rgb, detector=detector, cfg=cfg)  # [T,50,100,3]

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
            except Exception as e:
                n_fail += 1
                if n_fail <= 10:
                    print(f"[paper_preproc][FAIL] {video_path} err={e}", flush=True)
            finally:
                n_seen += 1
                if progress_every > 0 and (n_seen % int(progress_every) == 0):
                    dt = max(1e-6, time.time() - t0)
                    rate = n_seen / dt
                    print(
                        f"[paper_preproc] seen={n_seen}/{len(utterances)} "
                        f"train={n_train} val={n_val} test={n_test} skip={n_skip} fail={n_fail} "
                        f"rate={rate:.2f}/s",
                        flush=True,
                    )
    finally:
        for w in writers_train + writers_val + writers_test:
            w.close()

    elapsed = time.time() - t0
    meta = {
        "pipeline": "paper_style_dlib_ibug68_kalman_affine",
        "train_examples": n_train,
        "val_examples": n_val,
        "test_examples": n_test,
        "failed_examples": n_fail,
        "skipped_examples": n_skip,
        "seen_examples": n_seen,
        "elapsed_sec": elapsed,
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
        "input_root": str(input_root),
        "mode": split.mode,
        "val_speakers": sorted(list(split.val_speakers)),
        "test_speakers": sorted(list(split.test_speakers)),
        "exclude_speakers": sorted(list(split.exclude_speakers)),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta

