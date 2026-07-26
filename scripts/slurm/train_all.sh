#!/bin/bash
#SBATCH --job-name=mini-kanon3-embed
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

# Override these at submission time when your cluster paths differ:
#   sbatch --export=ALL,PROJECT_DIR=/path/to/min_Kano,CONDA_ENV=minKvenv scripts/slurm/train_all_embed.slurm
PROJECT_DIR="${PROJECT_DIR:-/fs04/scratch2/vf38/sloo0021/min_Kano}"
CONDA_ENV="${CONDA_ENV:-minKvenv}"

cd "$PROJECT_DIR"

module load miniforge3
mamba activate "$CONDA_ENV"

export PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-${SCRATCH:-$PROJECT_DIR/.cache}/huggingface}"
export WANDB_DIR="${WANDB_DIR:-${SCRATCH:-$PROJECT_DIR/.cache}/wandb}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
mkdir -p "$HF_HOME" "$WANDB_DIR"

python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable in this SLURM allocation")
print("GPU:", torch.cuda.get_device_name(0))
PY

for required in \
  data/processed/embed/train/queries.jsonl \
  data/processed/embed/validation/queries.jsonl \
  data/processed/embed/test/queries.jsonl; do
  [[ -f "$required" ]] || { echo "Missing required data: $required" >&2; exit 1; }
done

echo "===== Embed v1: in-batch negatives ====="
python src/training/main.py --config src/configs/embed/train.yaml

echo "===== Embed v2: BM25 hard negatives ====="
python scripts/mining/mine_bm25_negatives.py --config src/configs/embed/mine_bm25.yaml
python scripts/train/train_embed.py --config configs/embed/train_v2_bm25.yaml

echo "===== Embed v3: dense hard negatives ====="
python scripts/mining/mine_dense_negatives.py --config configs/embed/mine_dense.yaml
python scripts/train/train_embed.py --config configs/embed/train_v3_dense.yaml

echo "===== Embed v4: teacher distillation ====="
python scripts/mining/score_teacher_candidates.py --config configs/embed/score_teacher.yaml
python scripts/train/train_embed_distillation.py --config configs/embed/train_v4_distillation.yaml

echo "All Embed stages completed successfully."
