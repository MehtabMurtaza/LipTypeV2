## LipType TF2 rebuild (2026)

This folder contains a **clean TensorFlow 2 / Keras rebuild** of the LipType system described in the CHI'21 paper:
**“LipType: A Silent Speech Recognizer Augmented with an Independent Repair Model”**.

The original repo code under `LipType/`, `preprcoessing/`, and `postprocessing/` is kept for reference only (TF1-era).

### Components
- **Preprocessing**: mouth ROI extraction from videos + dataset conversion to TFRecords.
- **Recognizer (LipType)**: shallow 3D conv + SE-ResNet34 (time-distributed) + BiGRU×2 + CTC.
- **Repair model**:
  - **Light enhancement** (GLADNet-like) for low-light frames (optional).
  - **Postprocessing**: DDA denoiser + bidirectional trigram LM + edit-distance correction (optional).

### Quick start (after you prepare an environment)
- Install (editable):

```bash
python -m pip install -e .
```

- Show commands:

```bash
liptype2 --help
```

### Environments (recommended setup)

This repo can be run with **two conda envs**:

- **Training env**: `LipType/.venv311`
  - Used for: training (`train liptype`), evaluation (`eval liptype`), running scripts in `scripts/`.
- **Preprocess env (dlib)**: `LipType/.venv_dlib`
  - Used for: TFRecord conversion with `--dlib-predictor`.
  - Reason: `dlib` works reliably with **NumPy 1.26.x**, while some other packages in the training env may require **NumPy 2.x**.

Activate:

```bash
conda activate "/Users/mehtab/Documents/Research/LipType/.venv311"
# or
conda activate "/Users/mehtab/Documents/Research/LipType/.venv_dlib"
```

Run from repo root:

```bash
cd "/Users/mehtab/Documents/Research/LipType"
```

Tip: If `liptype2` isn’t on your PATH, you can always use:

```bash
python -m liptype_rebuild.cli.entrypoint --help
```

### End-to-end flow (GRID-style)

1) **Convert your existing dataset to TFRecords**

Your repo already contains GRID-like data at `data/s*_processed/`:
- videos: `data/s1_processed/*.mpg`
- alignments: `data/s1_processed/align/*.align`

Create a split file (example at `configs/grid_split_example.yaml`) and run:

```bash
python -m liptype_rebuild.cli.entrypoint preprocess grid-to-tfrecords \
  --input-root data \
  --output-root rebuild_data/tfrecords \
  --split-config configs/grid_split_example.yaml \
  --num-shards 64 \
  --max-frames 75 \
  --dlib-predictor /path/to/shape_predictor_68_face_landmarks.dat
```

If preprocessing is too slow (full GRID), create a smaller TFRecords set for testing:

```bash
python -m liptype_rebuild.cli.entrypoint preprocess grid-to-tfrecords \
  --input-root data \
  --output-root rebuild_data/tfrecords_small1 \
  --split-config configs/grid_split_example.yaml \
  --num-shards 8 \
  --max-frames 75 \
  --max-examples 200 \
  --dlib-predictor /path/to/shape_predictor_68_face_landmarks.dat
```

Notes:
- `--max-examples` stops after N utterances (total across all speakers).
- For dlib landmarks, run the converter inside the **dlib preprocess env** (`.venv_dlib`).

2) **Train LipType**

Edit `configs/liptype_grid.yaml` to point to your TFRecord shards, then:

```bash
python -m liptype_rebuild.cli.entrypoint train liptype \
  --config configs/liptype_grid.yaml \
  --run-dir runs/liptype_run1
```

3) **Evaluate**

```bash
python -m liptype_rebuild.cli.entrypoint eval liptype \
  --config configs/liptype_grid.yaml \
  --weights runs/liptype_run1/weights.001.weights.h5 \
  --num-batches 200
```

4) **Predict from a video**

```bash
python -m liptype_rebuild.cli.entrypoint predict liptype \
  --weights runs/liptype_run1/weights.001.weights.h5 \
  --video path/to/sample.mpg
```

If you want dlib-based mouth crops during prediction too:

```bash
python -m liptype_rebuild.cli.entrypoint predict liptype \
  --weights runs/liptype_run1/weights.001.weights.h5 \
  --video path/to/sample.mpg \
  --dlib-predictor /path/to/shape_predictor_68_face_landmarks.dat
```

### Qualitative check: print REF/HYP/WER from random TFRecord samples

After training (you have a `runs/.../weights.###.weights.h5`), you can sample random TFRecord examples and print:
- speaker/utterance id
- REF (ground truth from TFRecords)
- HYP (model output)
- WER

```bash
python scripts/random_val_decode.py \
  --config configs/liptype_grid.yaml \
  --weights runs/liptype_run1/weights.001.weights.h5 \
  --split val \
  --num-samples 10 \
  --beam-width 50
```

If your validation split is empty (e.g. very small TFRecords), use:

```bash
python scripts/random_val_decode.py \
  --config configs/liptype_grid.yaml \
  --weights runs/liptype_run1/weights.001.weights.h5 \
  --split train \
  --num-samples 10 \
  --beam-width 50
```

### TensorBoard

Training writes logs to `runs/<run_dir>/tb/`:

```bash
tensorboard --logdir runs/liptype_run1/tb
```

### Optional: low-light enhancement (GLADNet-like)

- Train from paired `low/*.png` and `normal/*.png`:

```bash
python -m liptype_rebuild.cli.entrypoint train gladnet --config configs/gladnet_example.yaml --run-dir runs/gladnet_run1
```

- Enhance a video:

```bash
python -m liptype_rebuild.cli.entrypoint enhance video \
  --weights runs/gladnet_run1/weights.001.weights.h5 \
  --input-video in.mp4 \
  --output-video out.mp4
```

### Optional: postprocessing repair model

The repair model can be applied at prediction time if you provide:
- a saved bidirectional trigram LM (`train lm`)
- a dictionary word list (one word per line)
- optionally a corpus text for Norvig spell correction

```bash
python -m liptype_rebuild.cli.entrypoint train lm \
  --corpus-txt LM-corpus.txt \
  --output-json runs/lm.json \
  --min-count 2

python -m liptype_rebuild.cli.entrypoint predict liptype \
  --weights runs/liptype_run1/weights.001.weights.h5 \
  --video path/to/sample.mpg \
  --repair-lm runs/lm.json \
  --repair-dict wordlist.txt
```

### Notes on naming
This rebuild is installed as the Python package `liptype_rebuild` and CLI `liptype2` to avoid case-insensitive import collisions with the legacy `LipType/` directory.

