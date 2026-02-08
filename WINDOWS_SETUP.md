## Windows setup guide (LipType TF2 rebuild)

This guide is for a **fresh Windows machine** after you `git pull` this repo.

The repo is easiest to run with **two conda environments**:

- **Training env**: for `train`, `eval`, `predict`, and scripts in `scripts/`
- **Preprocess env (dlib)**: for TFRecord conversion using `--dlib-predictor`

You can also skip the preprocess env if you copy TFRecords from your Mac and only want to train/evaluate on Windows.

---

### Prerequisites

- Install **Git**.
- Install **Miniconda** (recommended) or Anaconda.
- Open a terminal (PowerShell or Anaconda Prompt).

Repo location examples below assume:

```text
C:\Users\<you>\LipType
```

---

### 1) Get the code

```powershell
git clone <your-repo-url>
cd LipType
```

Or if it already exists:

```powershell
cd LipType
git pull
```

---

### 2) Training environment (recommended)

Create and activate:

```powershell
conda create -n liptype_train -y python=3.11
conda activate liptype_train
python -m pip install --upgrade pip
```

Install TensorFlow and project deps (CPU training is the simplest on native Windows):

```powershell
python -m pip install "tensorflow==2.20.*"
python -m pip install -e .
```

Quick sanity checks:

```powershell
python -c "import numpy as np; import tensorflow as tf; print('numpy', np.__version__, 'tf', tf.__version__)"
python -m liptype_rebuild.cli.entrypoint --help
```

Notes:
- If you have an NVIDIA GPU and want GPU training, the most reliable route is usually **WSL2 + Ubuntu**. Native Windows GPU support can be version-dependent.

---

### 3) Preprocessing environment (dlib + NumPy 1.26)

This environment is only needed if you want to generate TFRecords **on Windows** with **true dlib mouth landmarks**.

Create and activate:

```powershell
conda create -n liptype_dlib -y -c conda-forge python=3.11 numpy=1.26 dlib opencv pyyaml tqdm typer
conda activate liptype_dlib
python -m pip install --upgrade pip
```

Install TensorFlow for TFRecord writing + install this repo (editable, without pulling extra deps):

```powershell
python -m pip install "tensorflow==2.18.*"
python -m pip install -e . --no-deps
```

Sanity checks:

```powershell
python -c "import numpy as np, dlib, tensorflow as tf; print('numpy', np.__version__); print('dlib', dlib.__version__); print('tf', tf.__version__)"
```

Why a separate env?
- `dlib` is most stable with **NumPy 1.26.x**
- other packages (often `opencv-python` via pip) may pull **NumPy 2.x**

---

### 4) Download the dlib predictor file

You need `shape_predictor_68_face_landmarks.dat`.

Recommended: download `shape_predictor_68_face_landmarks.dat.bz2` and decompress it.

Create the folder:

```powershell
mkdir assets\dlib
```

Download (PowerShell uses `curl.exe`):

```powershell
curl.exe -L "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2" -o "assets\dlib\shape_predictor_68_face_landmarks.dat.bz2"
```

Decompress using Python (works everywhere, no extra tools):

```powershell
python - <<'PY'
import bz2
from pathlib import Path

inp = Path("assets/dlib/shape_predictor_68_face_landmarks.dat.bz2")
out = Path("assets/dlib/shape_predictor_68_face_landmarks.dat")
out.write_bytes(bz2.decompress(inp.read_bytes()))
print("wrote", out)
PY
```

---

### 5) Create TFRecords (small test set)

Run this in the **dlib preprocess env** (`liptype_dlib`):

```powershell
conda activate liptype_dlib
cd C:\Users\<you>\LipType
```

Generate a small TFRecords set (fast):

```powershell
python -m liptype_rebuild.cli.entrypoint preprocess grid-to-tfrecords `
  --input-root data `
  --output-root rebuild_data\tfrecords_small1 `
  --split-config configs\grid_split_example.yaml `
  --num-shards 8 `
  --max-frames 75 `
  --max-examples 200 `
  --dlib-predictor assets\dlib\shape_predictor_68_face_landmarks.dat
```

For a full dataset run, remove `--max-examples` and increase `--num-shards`:

```powershell
python -m liptype_rebuild.cli.entrypoint preprocess grid-to-tfrecords `
  --input-root data `
  --output-root rebuild_data\tfrecords_full_dlib `
  --split-config configs\grid_split_example.yaml `
  --num-shards 64 `
  --max-frames 75 `
  --dlib-predictor assets\dlib\shape_predictor_68_face_landmarks.dat
```

---

### 6) Train LipType

Run this in the **training env** (`liptype_train`):

```powershell
conda activate liptype_train
cd C:\Users\<you>\LipType
```

Point your YAML at your TFRecords (example):

```yaml
dataset:
  tfrecords_train: "rebuild_data/tfrecords_small1/train-*.tfrecord"
  tfrecords_val: "rebuild_data/tfrecords_small1/val-*.tfrecord"
```

Train:

```powershell
python -m liptype_rebuild.cli.entrypoint train liptype --config configs\liptype_train.yaml --run-dir runs\liptype_win_1
```

Weights will be written like:

```text
runs\liptype_win_1\weights.001.weights.h5
```

---

### 7) Evaluate statistics (WER)

```powershell
python -m liptype_rebuild.cli.entrypoint eval liptype `
  --config configs\liptype_train.yaml `
  --weights runs\liptype_win_1\weights.001.weights.h5 `
  --num-batches 200
```

---

### 8) Qualitative test: print REF/HYP/WER on random TFRecord samples

```powershell
python scripts\random_val_decode.py `
  --config configs\liptype_train.yaml `
  --weights runs\liptype_win_1\weights.001.weights.h5 `
  --split train `
  --num-samples 10 `
  --beam-width 50
```

Use `--split val` if your `val-*.tfrecord` actually has examples.

---

### Common issues

- **`python -m liptype_rebuild.cli.entrypoint ...` prints nothing**
  - Make sure you’re on the latest code. The entrypoint must include the `if __name__ == "__main__": main()` guard.

- **dlib errors like “Unsupported image type …”**
  - Use the **separate preprocess env** pinned to `numpy=1.26` (`liptype_dlib`).

- **Slow preprocessing**
  - Use `--max-examples` and fewer shards (e.g. `--num-shards 8`) to create `tfrecords_small1` for iteration.

