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

### End-to-end flow (GRID-style)

1) **Convert your existing dataset to TFRecords**

Your repo already contains GRID-like data at `data/s*_processed/`:
- videos: `data/s1_processed/*.mpg`
- alignments: `data/s1_processed/align/*.align`

Create a split file (example at `configs/grid_split_example.yaml`) and run:

```bash
liptype2 preprocess grid-to-tfrecords \
  --input-root data \
  --output-root rebuild_data/tfrecords \
  --split-config configs/grid_split_example.yaml \
  --num-shards 64 \
  --max-frames 75 \
  --dlib-predictor /path/to/shape_predictor_68_face_landmarks.dat
```

2) **Train LipType**

Edit `configs/liptype_grid.yaml` to point to your TFRecord shards, then:

```bash
liptype2 train liptype --config configs/liptype_grid.yaml --run-dir runs/liptype_run1
```

3) **Evaluate**

```bash
liptype2 eval liptype --config configs/liptype_grid.yaml --weights runs/liptype_run1/weights.001.weights.h5
```

4) **Predict from a video**

```bash
liptype2 predict liptype --weights runs/liptype_run1/weights.001.weights.h5 --video path/to/sample.mpg

If you want dlib-based mouth crops during prediction too:

```bash
liptype2 predict liptype \
  --weights runs/liptype_run1/weights.001.weights.h5 \
  --video path/to/sample.mpg \
  --dlib-predictor /path/to/shape_predictor_68_face_landmarks.dat
```
```

### Optional: low-light enhancement (GLADNet-like)

- Train from paired `low/*.png` and `normal/*.png`:

```bash
liptype2 train gladnet --config configs/gladnet_example.yaml --run-dir runs/gladnet_run1
```

- Enhance a video:

```bash
liptype2 enhance video --weights runs/gladnet_run1/weights.001.weights.h5 --input-video in.mp4 --output-video out.mp4
```

### Optional: postprocessing repair model

The repair model can be applied at prediction time if you provide:\n+- a saved bidirectional trigram LM (`liptype2 train lm ...`)\n+- a dictionary word list (one word per line)\n+- optionally a corpus text for Norvig spell correction

```bash
liptype2 train lm --corpus-txt LM-corpus.txt --output-json runs/lm.json --min-count 2

liptype2 predict liptype \
  --weights runs/liptype_run1/weights.001.weights.h5 \
  --video path/to/sample.mpg \
  --repair-lm runs/lm.json \
  --repair-dict wordlist.txt
```

### Notes on naming
This rebuild is installed as the Python package `liptype_rebuild` and CLI `liptype2` to avoid case-insensitive import collisions with the legacy `LipType/` directory.

