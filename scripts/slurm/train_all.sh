#!/bin/bash
#SBATCH --job-name=auslegal-embed-all
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/fs04/scratch2/vf38/sloo0021/legal_embedding}"
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
print(
    "GPU memory:",
    round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),
    "GiB",
)
PY

for required in \
  data/processed/embed/train/queries.jsonl \
  data/processed/embed/train/corpus.jsonl \
  data/processed/embed/train/qrels.tsv \
  data/processed/embed/validation/queries.jsonl \
  data/processed/embed/validation/corpus.jsonl \
  data/processed/embed/validation/qrels.tsv \
  data/processed/embed/test/queries.jsonl \
  data/processed/embed/test/corpus.jsonl \
  data/processed/embed/test/qrels.tsv; do

  if [[ ! -f "$required" ]]; then
    echo "Missing required data: $required" >&2
    exit 1
  fi
done


echo "============================================================"
echo "V1: in-batch-negative training"
echo "============================================================"

python src/training/main.py \
  --config src/configs/embed/train.yaml


echo "============================================================"
echo "Mine BM25 hard negatives"
echo "============================================================"

python src/mining/mine.py \
  --config src/configs/embed/mine_bm25.yaml

test -s data/interim/embed/negatives/bm25/train.jsonl || {
  echo "BM25 mining output is missing or empty" >&2
  exit 1
}


echo "============================================================"
echo "V2: BM25-hard-negative training"
echo "============================================================"

python src/training/main.py \
  --config src/configs/embed/train_v2_bm25.yaml


echo "============================================================"
echo "Mine dense hard negatives using V2"
echo "============================================================"

test -f artifacts/checkpoints/embed/gte-modernbert-base-v2-bm25/final/modules.json || {
  echo "V2 final checkpoint is missing" >&2
  exit 1
}

python src/mining/mine.py \
  --config src/configs/embed/mine_dense.yaml

test -s data/interim/embed/negatives/dense/train.jsonl || {
  echo "Dense mining output is missing or empty" >&2
  exit 1
}


echo "============================================================"
echo "V3: dense-hard-negative training"
echo "============================================================"

python src/training/main.py \
  --config src/configs/embed/train_v3_dense.yaml


echo "============================================================"
echo "Score candidates with cross-encoder teacher"
echo "============================================================"

python src/mining/mine.py \
  --config src/configs/embed/score_teacher.yaml

test -s data/interim/embed/teacher/cross_encoder/train.jsonl || {
  echo "Teacher-scoring output is missing or empty" >&2
  exit 1
}


echo "============================================================"
echo "V4: teacher-distillation training"
echo "============================================================"

python src/training/distill.py \
  --config src/configs/embed/train_v4_distillation.yaml


echo "============================================================"
echo "Verify final checkpoints"
echo "============================================================"

for checkpoint in \
  artifacts/checkpoints/embed/gte-modernbert-base/final \
  artifacts/checkpoints/embed/gte-modernbert-base-v2-bm25/final \
  artifacts/checkpoints/embed/gte-modernbert-base-v3-dense/final \
  artifacts/checkpoints/embed/gte-modernbert-base-v4-distillation/final; do

  if [[ ! -f "$checkpoint/modules.json" ]]; then
    echo "Missing completed checkpoint: $checkpoint" >&2
    exit 1
  fi

  echo "Found: $checkpoint"
done

echo "All Embed stages completed successfully."