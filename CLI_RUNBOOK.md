# LipTypeV2 (TF2 rebuild) — CLI runbook (Windows)

This repo contains:
- **Rebuild (TF2/Keras)**: `src/liptype_rebuild/` (use this)
- **Legacy (TF1-era)**: `LipType/`, `preprcoessing/`, `postprocessing/` (reference only)

The rebuild CLI can be run either as:
- `python -m liptype_rebuild.cli.entrypoint ...`
- or `liptype2 ...` (after `pip install -e .`)

---

## Paths (this workspace)
- **Repo root**: `C:\Mehtab_work\project_liptype\LipTypeV2`
- **GRID-style data root**: `C:\Mehtab_work\project_liptype\LipTypeV2\data`
  - expects `data\sX_processed\*.mpg` and `data\sX_processed\align\*.align`
- **dlib predictor**: `assets\dlib\shape_predictor_68_face_landmarks.dat`

---

## Environment 1: TFRecords preprocessing (parallel + dlib)

Create env (recommended for dlib stability with NumPy 1.26):

```powershell
cd C:\Mehtab_work\project_liptype\LipTypeV2
conda create -n liptype_dlib -y -c conda-forge python=3.11 numpy=1.26 dlib opencv pyyaml tqdm typer
conda activate liptype_dlib
python -m pip install --upgrade pip
python -m pip install "tensorflow==2.18.*"
python -m pip install -e . --no-deps
```

Download + decompress predictor:

```powershell
cd C:\Mehtab_work\project_liptype\LipTypeV2
mkdir assets\dlib
curl.exe -L "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2" -o "assets\dlib\shape_predictor_68_face_landmarks.dat.bz2"
python -c "import bz2, pathlib; inp=pathlib.Path(r'assets\dlib\shape_predictor_68_face_landmarks.dat.bz2'); out=pathlib.Path(r'assets\dlib\shape_predictor_68_face_landmarks.dat'); out.write_bytes(bz2.decompress(inp.read_bytes())); print('wrote', out)"
```

---

## Generate TFRecords (parallel) — speaker holdout split (train/val/test supported)

> Output will include `train-*`, `val-*`, and `test-*` shards.

```powershell
conda activate liptype_dlib
cd C:\Mehtab_work\project_liptype\LipTypeV2

python scripts\grid_to_tfrecords_parallel.py grid-to-tfrecords-parallel `
  --input-root "C:\Mehtab_work\project_liptype\LipTypeV2\data" `
  --output-root "C:\Mehtab_work\project_liptype\LipTypeV2\rebuild_data\tfrecords_parallel_speaker_holdout_dlib" `
  --split-config "C:\Mehtab_work\project_liptype\LipTypeV2\configs\grid_split_80_10_10_random42.yaml" `
  --num-shards 64 `
  --max-frames 75 `
  --workers 8 `
  --dlib-predictor "C:\Mehtab_work\project_liptype\LipTypeV2\assets\dlib\shape_predictor_68_face_landmarks.dat"
```

Smoke run (cap examples):

```powershell
python scripts\grid_to_tfrecords_parallel.py grid-to-tfrecords-parallel `
  --input-root "C:\Mehtab_work\project_liptype\LipTypeV2\data" `
  --output-root "C:\Mehtab_work\project_liptype\LipTypeV2\rebuild_data\tfrecords_parallel_smoke200_dlib" `
  --split-config "C:\Mehtab_work\project_liptype\LipTypeV2\configs\grid_split_80_10_10_random42.yaml" `
  --num-shards 64 `
  --max-frames 75 `
  --max-examples 200 `
  --workers 8 `
  --dlib-predictor "C:\Mehtab_work\project_liptype\LipTypeV2\assets\dlib\shape_predictor_68_face_landmarks.dat"
```

---

## Generate TFRecords (parallel) — paper-style overlapped validation + heldout test speakers

This matches:
- exclude `s21`
- **test speakers**: `s1,s2,s20,s22`
- **val**: 255 random utterances per remaining speaker
- **train**: all remaining utterances

Use:
- `configs\grid_split_overlapped_255_holdout_1_2_20_22.yaml`

```powershell
conda activate liptype_dlib
cd C:\Mehtab_work\project_liptype\LipTypeV2

python scripts\grid_to_tfrecords_parallel.py grid-to-tfrecords-parallel `
  --input-root "C:\Mehtab_work\project_liptype\LipTypeV2\data" `
  --output-root "C:\Mehtab_work\project_liptype\LipTypeV2\rebuild_data\tfrecords_overlapped255_test_1_2_20_22_dlib" `
  --split-config "C:\Mehtab_work\project_liptype\LipTypeV2\configs\grid_split_overlapped_255_holdout_1_2_20_22.yaml" `
  --num-shards 64 `
  --max-frames 75 `
  --workers 8 `
  --dlib-predictor "C:\Mehtab_work\project_liptype\LipTypeV2\assets\dlib\shape_predictor_68_face_landmarks.dat"
```

---

## Environment 2: Training (native Windows GPU, TF 2.10)

This is the native-Windows CUDA route that worked for RTX 3060 in this repo.

```powershell
cd C:\Mehtab_work\project_liptype\LipTypeV2
conda create -n liptype_win_cuda_clean -y python=3.10
conda activate liptype_win_cuda_clean
python -m pip install --upgrade pip

python -m pip install "numpy==1.23.5"
python -m pip install "tensorflow==2.10.1"
conda install -y -c conda-forge cudatoolkit=11.2 cudnn=8.1

python -m pip install pyyaml tqdm typer
python -m pip install -e . --no-deps

python -c "import numpy as np, tensorflow as tf; print('numpy', np.__version__); print('tf', tf.__version__); print('GPUs', tf.config.list_physical_devices('GPU'))"
```

Important:
- If you train with TF 2.10, **evaluate with the same env**. Loading these weights in TF 2.20/Keras 3 can fail.

---

## Train LipType

1) Update your YAML (example `configs\liptype_train.yaml`) to point at TFRecords:

```yaml
dataset:
  tfrecords_train: "rebuild_data/tfrecords_overlapped255_test_1_2_20_22_dlib/train-*.tfrecord"
  tfrecords_val: "rebuild_data/tfrecords_overlapped255_test_1_2_20_22_dlib/val-*.tfrecord"
  tfrecords_test: "rebuild_data/tfrecords_overlapped255_test_1_2_20_22_dlib/test-*.tfrecord"
```

2) Train:

```powershell
conda activate liptype_win_cuda_clean
cd C:\Mehtab_work\project_liptype\LipTypeV2

python -m liptype_rebuild.cli.entrypoint train liptype --config configs\liptype_train.yaml --run-dir runs\liptype_overlapped
```

Weights are saved like:
- `runs\liptype_overlapped\weights.001.weights.h5`

Note:
- The input pipeline filters examples that violate CTC constraints (e.g., very short/failed decode).

---

## Resume training from a weights checkpoint

This resumes **model weights** and continues at the next epoch number, but does **not** restore optimizer state.

```powershell
python -m liptype_rebuild.cli.entrypoint train liptype `
  --config configs\liptype_train.yaml `
  --run-dir runs\liptype_overlapped `
  --resume-weights runs\liptype_overlapped\weights.019.weights.h5
```

### Resume training with optimizer state (true resume)

Training also saves a TensorFlow checkpoint each epoch under:
- `runs\<run_dir>\ckpt\`

This restores:
- model weights
- optimizer state (e.g. Adam moments)
- epoch counter

Resume from the latest checkpoint in the run dir:

```powershell
python -m liptype_rebuild.cli.entrypoint train liptype `
  --config configs\liptype_train.yaml `
  --run-dir runs\liptype_overlapped `
  --resume-checkpoint runs\liptype_overlapped\ckpt
```

---

## Evaluate (WER) on val/train/test

```powershell
conda activate liptype_win_cuda_clean
cd C:\Mehtab_work\project_liptype\LipTypeV2

python -m liptype_rebuild.cli.entrypoint eval liptype `
  --config configs\liptype_train.yaml `
  --weights runs\liptype_overlapped\weights.019.weights.h5 `
  --split val `
  --num-batches 200
```

Evaluate on **test**:

```powershell
python -m liptype_rebuild.cli.entrypoint eval liptype `
  --config configs\liptype_train.yaml `
  --weights runs\liptype_overlapped\weights.019.weights.h5 `
  --split test `
  --num-batches 200
```

---

## Qualitative decode (print REF/HYP/WER) — train/val/test

```powershell
conda activate liptype_win_cuda_clean
cd C:\Mehtab_work\project_liptype\LipTypeV2

python scripts\random_val_decode.py `
  --config configs\liptype_train.yaml `
  --weights runs\liptype_overlapped\weights.019.weights.h5 `
  --split val `
  --num-samples 10 `
  --beam-width 50
```

Use `--split test` to sample from the held-out speakers.

---

## Predict from a single video (optional)

`predict liptype` crops mouth ROIs and needs landmarking dependencies (dlib or mediapipe backend). If you want to do this, use the TF2.20 env (`liptype_train`) or create a dedicated predict env; avoid mixing it into the TF 2.10 training env.

Example:

```powershell
python -m liptype_rebuild.cli.entrypoint predict liptype `
  --weights runs\liptype_overlapped\weights.019.weights.h5 `
  --video "C:\Mehtab_work\project_liptype\LipTypeV2\data\s1_processed\bbaf2n.mpg" `
  --beam-width 50 `
  --dlib-predictor "C:\Mehtab_work\project_liptype\LipTypeV2\assets\dlib\shape_predictor_68_face_landmarks.dat"
```

