# Parallel TFRecord preprocessing (GRID-style)

This repo’s default TFRecord converter (`liptype2 preprocess grid-to-tfrecords`) is **single-process**. Landmark detection (especially dlib) is CPU-heavy, so preprocessing can take many hours.

To speed this up **without modifying the existing code**, we added a separate script:

- `scripts/grid_to_tfrecords_parallel.py`

It uses **multiprocessing** and writes the **same TFRecord filename layout** as the original converter, so training configs can stay the same.

---

## What was added

### 1) Parallel TFRecord generator

- **File**: `scripts/grid_to_tfrecords_parallel.py`
- **Command**: `grid-to-tfrecords-parallel`
- **Parallelization strategy**:
  - Enumerate all utterances once (stable order).
  - Assign shard by the same rule as the existing converter:
    - `shard = idx % num_shards`
  - Run **one process per shard** (bounded by `--workers`).
  - Each worker writes **only its own shard files**, so there’s no concurrent writing to the same TFRecord.

### 2) Helper command to sanity-check discovery

- **Command**: `count-utterances`
- Prints how many `(video, align)` pairs are found under `--input-root` using the same discovery logic as preprocessing.

### 3) Example 80/10/10 speaker split (train/val/test)

- **File**: `configs/grid_split_80_10_10_random42.yaml`
- This is a **speaker-level split** (random seed 42) over the speakers present in `data/s*_processed/`.
- Note: current TFRecord writing logic uses only `train_speakers` + `val_speakers`; `test_speakers` is included for convenience/future use.

---

## How to run

### Count utterances (optional)

```bash
python scripts/grid_to_tfrecords_parallel.py count-utterances --input-root data
```

### Generate TFRecords in parallel (recommended)

```bash
python scripts/grid_to_tfrecords_parallel.py grid-to-tfrecords-parallel \
  --input-root data \
  --output-root rebuild_data/tfrecords_parallel_1 \
  --split-config configs/grid_split_example.yaml \
  --num-shards 64 \
  --max-frames 75 \
  --max-examples 0 \
  --workers 8 \
  --dlib-predictor assets/dlib/shape_predictor_68_face_landmarks.dat
```

### Use your new 80/10/10 speaker split

```bash
python scripts/grid_to_tfrecords_parallel.py grid-to-tfrecords-parallel \
  --input-root data \
  --output-root rebuild_data/tfrecords_parallel_80_10_10 \
  --split-config configs/grid_split_80_10_10_random42.yaml \
  --num-shards 64 \
  --max-frames 75 \
  --workers 8 \
  --dlib-predictor assets/dlib/shape_predictor_68_face_landmarks.dat
```

---

## Key options

- **`--workers`**: number of worker processes.
  - Start with **8**, then try **12** or **16** if the machine stays responsive and you don’t hit memory/thermal limits.
- **`--num-shards`**: how many shard files are produced *per split* (`train-*` and `val-*`).
  - Common values: **32**, **64**, **128**
  - More shards = more (smaller) files and more training I/O parallelism; fewer shards = fewer (larger) files.
  - This does **not** directly speed up preprocessing much; it mostly affects output layout and training input throughput.
- **`--max-frames`**: frame count stored per example (examples are padded/truncated to this length).
- **`--max-examples`**: if > 0, stops after N utterances (useful for smoke tests).
- **`--decode-max-frames`**:
  - Default `-1` means “decode up to `--max-frames` frames per video” (faster).
  - `0` means “decode the full video” (slower).

---

## Output layout

The script writes:

- `output_root/train-00000-of-00064.tfrecord` ... `train-00063-of-00064.tfrecord`
- `output_root/val-00000-of-00064.tfrecord` ... `val-00063-of-00064.tfrecord`
- `output_root/meta.json`

This matches the existing converter’s naming convention.

---

## Notes on decoder warnings

If you see messages like:

```
[mpeg1video] ac-tex damaged ...
[mpeg1video] Warning MVs not available
```

they come from the video decoder (FFmpeg/OpenCV). Usually preprocessing continues; at worst a small number of frames may decode with artifacts. Alignments come from `.align` text files and are independent of video decoding.

