## NVIDIA GPU setup notes (TensorFlow + this repo)

This document explains what typically needs to change to run training on an **NVIDIA GPU**.

The main point: **modern TensorFlow GPU support is best on Linux**.  
On Windows, the most reliable approach is **WSL2 (Ubuntu) + NVIDIA drivers**.

---

### Recommended options (in order)

1) **Linux** (native Ubuntu) + NVIDIA driver  
2) **Windows + WSL2 (Ubuntu)** + NVIDIA driver (recommended for Windows laptops/desktops)  
3) **Docker** on Linux/WSL2 using an NVIDIA-enabled container runtime

Native Windows (no WSL2) GPU installs are commonly fragile and, depending on your TF version, may be unsupported.

---

### A) WSL2 (Ubuntu) approach (recommended on Windows)

#### 1) Install prerequisites
- Install **WSL2** and Ubuntu (e.g. Ubuntu 22.04).
- Install the **Windows NVIDIA driver that supports WSL2 CUDA**.
- Inside Ubuntu, install a few basics:

```bash
sudo apt update
sudo apt install -y git python3 python3-venv python3-pip
```

#### 2) Clone repo in Ubuntu

```bash
git clone <your-repo-url>
cd LipType
```

#### 3) Create a training venv (GPU)

```bash
python3 -m venv .venv_gpu
source .venv_gpu/bin/activate
python -m pip install --upgrade pip
```

#### 4) Install TensorFlow with NVIDIA CUDA dependencies

For TensorFlow **2.20.x**, the simplest path on Linux/WSL2 is to let pip install the matching NVIDIA CUDA libs:

```bash
python -m pip install "tensorflow[and-cuda]==2.20.*"
python -m pip install -e .
```

Notes:
- The `and-cuda` extra is designed to pull the correct CUDA/cuDNN runtime packages automatically (no system-wide CUDA toolkit required in most cases).

#### 5) Verify GPU is visible

```bash
python -c "import tensorflow as tf; print(tf.__version__); print(tf.config.list_physical_devices('GPU'))"
```

If this prints a GPU device, you’re good.

#### 6) Train as usual

```bash
python -m liptype_rebuild.cli.entrypoint train liptype --config configs/liptype_train.yaml --run-dir runs/liptype_gpu_1
```

---

### B) Native Linux (Ubuntu) approach

Same as WSL2, except you install the standard Linux NVIDIA driver.

You can use either:
- `pip install "tensorflow[and-cuda]==2.20.*"` (recommended), or
- a system CUDA toolkit + cuDNN install (more work, more chances for version mismatch).

Verification is the same:

```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

---

### C) Docker approach (Linux/WSL2)

Use Docker with NVIDIA runtime:
- Install `nvidia-container-toolkit`
- Use a GPU-enabled TensorFlow image (or your own Dockerfile) and run with `--gpus all`.

This can be the most reproducible setup for multi-machine training.

---

### What (if anything) changes in this repo for GPU?

Usually **nothing** in the code. GPU vs CPU is decided by TensorFlow at runtime.

Things you *might* tune in `configs/liptype_train.yaml`:
- **`train.batch_size`**: increase until you hit GPU memory limits (common first knob)
- **`decode.beam_width`**: decoding is CPU-heavy; for faster epochs you can reduce it during training-time eval
- **`train.epochs`**: on GPU you’ll likely train longer since it’s faster

---

### Common GPU troubleshooting

- **GPU not detected**
  - On WSL2, verify you installed an NVIDIA driver with WSL2 CUDA support.
  - Run `nvidia-smi` (Linux) or the WSL2 CUDA checks.
  - Reinstall TF with `tensorflow[and-cuda]` inside a clean env.

- **CUDA/cuDNN version mismatch**
  - Prefer `tensorflow[and-cuda]` to avoid manual CUDA installs.

- **Out of memory**
  - Reduce `batch_size`
  - Reduce model size (advanced) or frame sizes (changes dataset/model assumptions)

