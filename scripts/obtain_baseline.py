# File:         scripts/obtain_baseline.py
# Author:       Lea Button
# Date:         25-09-2025
# Description:  Script to obtain zero-shot baseline predictions for a given 
#               model on a specified example from the BigVul dataset.

import os
import sys
import json
import argparse

from prompts.baseline_runner import BaselineRunner
from prompts.prompt_builder import PromptBuilder
from data.bigvul_loader import BigVulDataset
from scripts.utils import measured_phase, write_jsonl

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.environ["TRANSFORMERS_CACHE"] = "./cache/hf_cache"

MODEL_CONFIGS = {
    "starcoder2-3b": {
        "name": "bigcode/starcoder2-3b", 
        "tag" : "starcoder2_3b",
        "safetensors": True
        },
    "starcoder2-7b": {
        "name": "bigcode/starcoder2-7b", 
        "tag" : "starcoder2_7b",
        "safetensors": True
        },
    "tinyllama": {
        "name": "TinyLlama/TinyLlama_v1.1",
        "tag": "tinyllama"
        },
    "codellama-7b": {
        "name": "meta-llama/CodeLlama-7b-hf",
        "tag": "codellama_7b"
        },
    "phi-1.5": {
        "name": "microsoft/phi-1_5", 
        "tag": "phi_1_5",
        "safetensors": True
        },
    "deepseek-1.3b": {
        "name": "deepseek-ai/deepseek-coder-1.3b-base",
        "tag": "deepseek_coder_1_3b"
        },
    "deepseek-6.7b": {
        "name": "deepseek-ai/deepseek-coder-6.7b-base",
        "tag": "deepseek_coder_6_7b"
        },
}

def already_done(save_path, ex_id):
    """Check if an ID is already in the JSONL file."""
    if not os.path.exists(save_path):
        return False
    with open(save_path, "r") as f:
        for line in f:
            try:
                entry = json.loads(line)
                if entry.get("id") == ex_id:
                    return True
            except Exception:
                continue
    return False


def main():
    parser = argparse.ArgumentParser(description="Run Zero/One/Three-Shot Baseline (one example per job)")
    parser.add_argument("--model", required=True, choices=MODEL_CONFIGS.keys(), help="Model to run")
    parser.add_argument("--shots", type=int, default=0, help="Number of examples to include in the prompt (0 for zero-shot)")
    parser.add_argument("--csv_path", type=str, default="data/MSR_data_cleaned.csv")
    parser.add_argument("--index", type=int, required=True, help="Index of dataset example (HPC array style)")
    parser.add_argument("--measure_log", type=str, default="measure/baseline_logs.jsonl")
    args = parser.parse_args()

    config = MODEL_CONFIGS[args.model]
    builder = PromptBuilder(shots=args.shots)
    dataset = BigVulDataset(args.csv_path)
    
    if args.index < 0 or args.index >= len(dataset):
        print(f"Index {args.index} out of range (dataset size={len(dataset)}).")
        return
    ex = dataset[args.index]

    # Output path
    model_tag = config["tag"]
    save_path = f"results/{args.shots}_shot/{model_tag}_baseline_predictions.jsonl"
    
    # Skip if already exists
    if already_done(save_path, ex["id"]):
        print(f"Skipping {ex['id']} (already in {save_path}).")
        return

    # Build prompt
    prompt = builder.build_base_prompt(ex["code_numbered"])
    code_len = len(ex["code_numbered"].splitlines())

    logs = []

    # Run single example prediction (CPU only)
    runner = BaselineRunner(config["name"], safetensors=config.get("safetensors", False), batch_size=1, use_accelerate=False)
    
    with measured_phase("Baseline_LM_infer", logs, {
        "example_id": ex["id"],
        "model": config["name"],
        "shots": args.shots,
        "code_len": code_len
    }):
        output = runner.run_prompt_batch([prompt])[0]
    
    with measured_phase("Baseline_Postproc", logs, {
        "example_id": ex["id"],
        "model": config["name"],
        "shots": args.shots
    }):
        predicted = runner.extract_line_numbers(output)

    # Save
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "a") as f:
        json.dump({
            "id": ex["id"],
            "commit_id": ex["commit_id"],
            "true_lines": ex["vuln_lines"],
            "prompt": prompt,
            "output": output,
            "predicted_lines": predicted,
        }, f)
        f.write("\n")

    write_jsonl(args.measure_log, logs)
    print(f"Saved prediction for {ex['id']} to {save_path}")


if __name__ == "__main__":
    main()