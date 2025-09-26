#!/bin/bash
#SBATCH --job-name=rq4
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=${PARTITION:-cpu}
#SBATCH --gres=${GRES:-none}
#SBATCH --mail-user=lgbutton1@sheffield.ac.uk
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=logs/rq4_%x_%A_%a.out
#SBATCH --error=logs/rq4_%x_%A_%a.err

# Load necessary modules
module load Anaconda3/2024.02-1
module load GCC/11.2.0
source activate myspark

# Set variables
MODE=${MODE:?Set MODE=baseline_array|attn_array|train_cv|report}
MODEL_NAME=${MODEL_NAME:-"deepseek-ai/deepseek-coder-1.3b-base"}
MODEL_KEY=${MODEL_KEY:-"deepseek-1.3b"} 
SHOTS=${SHOTS:-0}
CSV_PATH=${CSV_PATH:-"data/MSR_data_cleaned.csv"}
CV_DIR=${CV_DIR:-"data/cv"}
ATTN_CACHE=${ATTN_CACHE:-"cache/attn_matrices"}
CLS_CACHE=${CLS_CACHE:-"cache/attn_classifiers"}
EPOCHS=${EPOCHS:-5}
BATCH_SIZE=${BATCH_SIZE:-16}
CLASSIFIER=${CLASSIFIER:-"BiLSTM_moderate"}

# Create necessary directories
LOG_ROOT=measure
mkdir -p "$LOG_ROOT" logs "$ATTN_CACHE" "$CLS_CACHE"

echo "[INFO] MODE=$MODE  SHOTS=$SHOTS  SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-NA}"


# Main logic
if [[ "$MODE" == "baseline_array" ]]; then
  # one example per array task (CPU)
  INDEX=$(( ${SLURM_ARRAY_TASK_ID:?need --array} + ${OFFSET:-0} ))
  python -m scripts.obtain_zsl_baseline \
    --model "$MODEL_KEY" \
    --shots "$SHOTS" \
    --csv_path "$CSV_PATH" \
    --index "$INDEX" \
    --measure_log "$LOG_ROOT/baseline_logs.jsonl"

elif [[ "$MODE" == "attn_array" ]]; then
  # one example per array task
  INDEX=$(( ${SLURM_ARRAY_TASK_ID:?need --array} + ${OFFSET:-0} ))
  python -m scripts.compute_attention \
    --csv_path "$CSV_PATH" \
    --index "$INDEX" \
    --model_name "$MODEL_NAME" \
    --shots "$SHOTS" \
    --output_dir "$ATTN_CACHE" \
    --measure_log "$LOG_ROOT/lova_precompute.jsonl"

elif [[ "$MODE" == "train_cv" ]]; then
  # single job
  python -m scripts.train_lova_cv \
    --model_name "$MODEL_NAME" \
    --cv_dir "$CV_DIR" \
    --output_dir "$CLS_CACHE" \
    --classifier "$CLASSIFIER" \
    --shots "$SHOTS" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --measure_log "$LOG_ROOT/lova_train_infer.jsonl"

else
  echo "Unknown MODE=$MODE"
  exit 1
fi
