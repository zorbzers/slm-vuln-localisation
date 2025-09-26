# File:         data/split_kfold_data.py
# Author:       Lea Button
# Date:         25-09-2025
# Description:  Create k-fold cross-validation splits of BigVul dataset

import os
import json
import random
from sklearn.model_selection import KFold
from data.bigvul_loader import BigVulDataset

DATA_PATH = "data/MSR_data_cleaned.csv"
OUTPUT_DIR = "data/cv"
N_SPLITS = 5
SEED = 240167697

# Load dataset
print("Loading dataset...")
dataset = BigVulDataset(DATA_PATH, load_attention=False)
dataset = list(dataset)
print(f"Loaded {len(dataset)} examples.")

# Shuffle for reproducibility
random.seed(SEED)
random.shuffle(dataset)

# Initialize KFold
n_total_folds = N_SPLITS + 1
kf = KFold(n_splits=n_total_folds, shuffle=False)

folds = []
for _, fold_idx in kf.split(dataset):
    folds.append([dataset[i] for i in fold_idx])

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

for fold_idx in range(N_SPLITS):
    fold_dir = os.path.join(OUTPUT_DIR, f"fold_{fold_idx}")
    os.makedirs(fold_dir, exist_ok=True)

    val_data = folds[fold_idx]
    train_data = [ex for j, fold in enumerate(folds) if j != fold_idx and j < N_SPLITS for ex in fold]

    with open(os.path.join(fold_dir, "train.json"), "w") as f:
        json.dump(train_data, f, indent=2)

    with open(os.path.join(fold_dir, "val.json"), "w") as f:
        json.dump(val_data, f, indent=2)

    print(f"Saved fold {fold_idx}: {len(train_data)} train, {len(val_data)} val")

# Test fold
test_fold = folds[N_SPLITS]
fold_dir = os.path.join(OUTPUT_DIR, f"fold_{N_SPLITS}")
os.makedirs(fold_dir, exist_ok=True)

with open(os.path.join(fold_dir, "test.json"), "w") as f:
    json.dump(test_fold, f, indent=2)

print(f"Saved test fold_{N_SPLITS}: {len(test_fold)} test examples")

print("Done.")

