# File:         scripts/train_lova_cv.py
# Author:       Lea Button
# Date:         25-09-2025
# Description:  Train and evaluate LOVA classifiers with different architectures

import os
import json
import torch
import argparse
from torch.utils.data import DataLoader, TensorDataset

from lova.lova import Lova
from data.attn_dataset import AttnDataset

from scripts.utils import measured_phase, write_jsonl, collate_fn


def compute_pos_weight(dataset):
    """Compute positive weighting for imbalanced BCE loss."""
    total_pos, total_neg = 0, 0
    for _, y, _ in dataset:
        total_pos += y.sum().item()
        total_neg += len(y) - y.sum().item()
    return (total_neg / total_pos) if total_pos > 0 else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cv_dir", type=str, default="data/cv")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--classifier", type=str, required=True)
    parser.add_argument("--shots", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=5, help="Epochs (torch models only)")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--output_dir", type=str, default="cache/attn_classifiers")
    parser.add_argument("--cache_dir", type=str, default="cache/attn_matrices")
    parser.add_argument("--measure_log", type=str, default="measure/lova_train_infer.jsonl")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = args.model_name.replace("/", "_")
    classifier_id = args.classifier
    base_out_dir = os.path.join(args.output_dir, model_id, f"{args.shots}_shot", classifier_id)
    os.makedirs(base_out_dir, exist_ok=True)

    summary = []
    all_logs = []

    for fold in range(5):
        print(f"\n==== Fold {fold} ({args.classifier}) ====")
        cv_dir = os.path.join(args.cv_dir, f"fold_{fold}")

        # Prepare datasets
        train_dataset = AttnDataset(
            os.path.join(cv_dir, "train.json"),
            args.model_name,
            args.cache_dir,
            shots=args.shots,
            device=device
        )
        val_dataset = AttnDataset(
            os.path.join(cv_dir, "val.json"),
            args.model_name,
            args.cache_dir,
            shots=args.shots,
            device=device
        )

        # Build DataLoaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            collate_fn=collate_fn,
        )

        # Initialise LOVA
        sample_X, _, _ = train_dataset[0]   # (X, y, metadata)
        input_dim = sample_X.shape[-1]
        
        lova = Lova(shots=args.shots, classifier_name=args.classifier, device=device)
        lova.initialize_classifier(input_dim=input_dim)

        # Handle imbalance
        pos_weight = compute_pos_weight(train_dataset)
        if pos_weight:
            print(f"Using pos_weight={pos_weight:.3f}")

        logs = []

        with measured_phase("LoVA_Train", logs, {
            "fold": fold, "classifier": args.classifier, "shots": args.shots,
            "epochs": args.epochs, "batch_size": args.batch_size
        }):
            history = []

            # Train + evaluate
            for log in lova.train_classifier(train_loader, epochs=args.epochs, pos_weight=pos_weight):
                metrics = lova.evaluate(val_loader)
                log.update(metrics)
                history.append(log)
                print(f"Epoch {log['epoch']} | Loss={log['train_loss']} "
                    f"| F1={metrics['f1']:.4f} | P={metrics['precision']:.4f} | R={metrics['recall']:.4f}")

        # Collect logs for this fold
        all_logs.extend(logs)

        # Tune threshold
        best_thresh, best_f1 = lova.tune_threshold(val_loader)
        print(f"Best threshold for fold {fold}: {best_thresh:.2f} (F1={best_f1:.4f})")

        # Save model
        model_ext = "pt" if isinstance(lova.classifier, torch.nn.Module) else "pkl"
        model_path = os.path.join(base_out_dir, f"fold_{fold}.{model_ext}")
        lova.save_classifier(model_path)

        # Save metrics
        metrics_path = os.path.join(base_out_dir, f"fold_{fold}_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(history, f, indent=2)

        summary.append(history[-1])

    # Save measurement logs across folds
    write_jsonl(args.measure_log, all_logs)

    # Save summary across folds
    summary_path = os.path.join(base_out_dir, "cv_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nAll folds complete. Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
