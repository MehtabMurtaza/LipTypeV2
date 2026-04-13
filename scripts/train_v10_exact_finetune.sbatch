#!/bin/bash
#SBATCH --job-name=liptype_v10_finetune
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH -t 3-00:00:00
#SBATCH -J liptype_gpu
#SBATCH -o exact_pipeline_logs/%x-%j.out
#SBATCH -e exact_pipeline_logs/%x-%j.err

set -euo pipefail

source /home/smurtaza/data/miniforge3/etc/profile.d/conda.sh
conda activate liptype_pinn_gpu

cd /home/smurtaza/data/LipTypeV2
mkdir -p exact_pipeline_logs

nvidia-smi
python -c "import tensorflow as tf; print('TF', tf.__version__, 'GPUs', tf.config.list_physical_devices('GPU'))"

python -m liptype_rebuild.cli.entrypoint train liptype \
  --config configs/liptype_v10_exact_finetune.yaml \
  --run-dir runs/liptype_v10_exact_finetune \
  --resume-weights runs/liptype_grid_paper_preproc/weights.001.weights.h5
