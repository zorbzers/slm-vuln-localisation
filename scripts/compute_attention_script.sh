#!/bin/bash
#SBATCH --job-name=lova_attn_compute
#SBATCH --time=00:45:00
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
OFFSET=${OFFSET:-0}
INDEX=$((SLURM_ARRAY_TASK_ID + OFFSET))

MODEL=${MODEL:?Error: MODEL not set. Pass with sbatch --export=MODEL=...}
SHOTS=${SHOTS:?Error: SHOTS not set. Pass with sbatch --export=SHOTS=...}

# Create log directory and redirect output
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
exec > >(tee "$LOG_DIR/lova_attn_calc_idx_${INDEX}.out")
exec 2> >(tee "$LOG_DIR/lova_attn_calc_idx_${INDEX}.err" >&2)
mkdir -p "cache/attn_matrices"

# Run the Python script
conda run -n myspark python -m scripts.compute_attention \
    --csv_path $CSV_PATH \
    --index $INDEX \
    --model_name $MODEL \
    --shots $SHOTS \
    --output_dir cache/attn_matrices
