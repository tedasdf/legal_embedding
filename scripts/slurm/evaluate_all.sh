#!/bin/bash
#SBATCH --job-name=auslegal-eval-all
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
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
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

SPLIT="data/processed/embed/validation"
OUTPUT="reports/experiments/embed/validation_all"

mkdir -p "$OUTPUT" "$HF_HOME"

MODELS=(
  "bm25"
  "Alibaba-NLP/gte-modernbert-base"
  "artifacts/checkpoints/embed/gte-modernbert-base/final"
  "artifacts/checkpoints/embed/gte-modernbert-base-v2-bm25/final"
  "artifacts/checkpoints/embed/gte-modernbert-base-v3-dense/final"
  "artifacts/checkpoints/embed/gte-modernbert-base-v4-distillation/final"
)

for MODEL in "${MODELS[@]}"; do
  echo "============================================================"
  echo "Evaluating: $MODEL"
  echo "============================================================"

  python src/evaluate/evaluate.py \
    --model "$MODEL" \
    --split "$SPLIT" \
    --output "$OUTPUT" \
    --batch-size 32 \
    --device cuda \
    --trust-remote-code
done

echo "All model evaluations completed successfully."
echo "Reports: $OUTPUT"