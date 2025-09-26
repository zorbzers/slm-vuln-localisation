#!/bin/bash
#SBATCH --job-name=train_attn_cv
#SBATCH --output=logs/train_attn_cv.out
#SBATCH --error=logs/train_attn_cv.err
#SBATCH --time=06:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --mail-user=lgbutton1@sheffield.ac.uk
#SBATCH --mail-type=BEGIN,END,FAIL 

# Load necessary modules
module load Anaconda3/2024.02-1
module load GCC/11.2.0

# Activate conda environment
source activate myspark

# Set variables
MODEL_NAME=${MODEL_NAME:-"deepseek-ai/deepseek-coder-1.3b-base"}
CV_DIR=${CV_DIR:-"data/cv"}
OUTPUT_DIR=${OUTPUT_DIR:-"cache/attn_classifiers"}
CLASSIFIER=${CLASSIFIER:-"BiLSTM_moderate"}
SHOTS=${SHOTS:-0}
EPOCHS=${EPOCHS:-5}
BATCH_SIZE=${BATCH_SIZE:-16}

echo "Running training with:"
echo "  MODEL_NAME = $MODEL_NAME"
echo "  CLASSIFIER = $CLASSIFIER"
echo "  SHOTS      = $SHOTS"
echo "  EPOCHS     = $EPOCHS"
echo "  BATCH_SIZE = $BATCH_SIZE"

# Run the Python script
python -m scripts.train_attn_cv \
  --model_name "$MODEL_NAME" \
  --cv_dir "$CV_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --classifier "$CLASSIFIER" \
  --shots $SHOTS \
  --epochs $EPOCHS \
  --batch_size $BATCH_SIZE
