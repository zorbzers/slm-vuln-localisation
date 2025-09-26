#!/bin/bash
#SBATCH --job-name=baseline_predictions
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=0-999
#SBATCH --mail-user=lgbutton1@sheffield.ac.uk
#SBATCH --mail-type=BEGIN,END,FAIL

# Load necessary modules
module load Anaconda3/2024.02-1
module load GCC/11.2.0

# Activate conda environment
source activate myspark

# Set variables
CSV_PATH="data/MSR_data_cleaned.csv"
INDEX=$((SLURM_ARRAY_TASK_ID + OFFSET))

MODEL=${MODEL:?Error: MODEL not set. Pass with sbatch --export=MODEL=...}
SHOTS=${SHOTS:?Error: SHOTS not set. Pass with sbatch --export=SHOTS=...}
OFFSET=${OFFSET:-0}

# Create log directory and redirect output
LOG_DIR="logs/${MODEL}/${SHOTS}_shot"
mkdir -p "$LOG_DIR"
exec > >(tee "$LOG_DIR/baseline_pred_idx_${INDEX}.out")
exec 2> >(tee "$LOG_DIR/baseline_pred_idx_${INDEX}.err" >&2)

# Run the Python script
python -m scripts.obtain_baseline \
    --model $MODEL \
    --csv_path $CSV_PATH \
    --index $INDEX \
    --shots $SHOTS 
