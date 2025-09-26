# File:     scripts/compute_attention.py
# Author:   Lea Button
# Date:     25-09-2025
# Description: Script to compute and cache LoVA attention matrices for examples in the BigVul dataset.

import torch
import os
import argparse

from transformers import AutoTokenizer, AutoModelForCausalLM
from lova.lova import Lova
from data.bigvul_loader import BigVulDataset
from scripts.utils import measured_phase, write_jsonl

def save_matrix(output_path, matrix, valid_lines):
    """Save the attention matrix and valid lines to a file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save({"attn_matrix": matrix, "valid_lines": valid_lines}, output_path)
    
    
def is_valid_matrix(path):
    """Check if a saved matrix file exists and is valid."""
    if not os.path.exists(path):
        return False
    try:
        data = torch.load(path, map_location="cpu")
        return isinstance(data, dict) and "attn_matrix" in data and "valid_lines" in data
    except Exception:
        return False
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, default="data/MSR_data_cleaned.csv")
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--shots", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="cache/attn_matrices")
    parser.add_argument("--measure_log", type=str, default="measure/lova_precompute.jsonl")
    args = parser.parse_args()

    dataset = BigVulDataset(args.csv_path)
    example = dataset[args.index]
    
    out_path = os.path.join(
        args.output_dir,
        args.model_name.replace("/", "_"),
        f"{args.shots}_shot",
        f"{example['id']}.pt"
    )
    
    # Skip if already exists
    if is_valid_matrix(out_path):
        print(f"Skipping {example['id']} (valid cache exists at {out_path})")
        exit(0)
    else:
        print(f"Recomputing {example['id']} (missing or corrupt cache)")
    
    torch.set_num_threads(48)

    model = AutoModelForCausalLM.from_pretrained(args.model_name,
                                                 torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                                                 device_map="auto" if torch.cuda.is_available() else None)
   
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    lova = Lova(model, tokenizer, device='cuda' if torch.cuda.is_available() else 'cpu', shots=args.shots)
    
    logs = []
    code_len = len(example["code_numbered"].splitlines())

    with measured_phase("LoVA_LM_infer+attn_extract", logs, {
        "example_id": example["id"],
        "model": args.model_name,
        "shots": args.shots,
        "code_len": code_len
    }):
        matrix, valid_lines = lova.extract_lova_matrix(example["code_numbered"], example["line_map"])
        
    out_path = os.path.join(args.output_dir, 
                            args.model_name.replace("/", "_"), 
                            f"{args.shots.__str__()}_shot",
                            f"{example['id']}.pt")
    
    with measured_phase("LoVA_Cache_write", logs, {
        "example_id": example["id"],
        "model": args.model_name,
        "shots": args.shots,
        "out_path": out_path
    }):
        save_matrix(out_path, matrix, valid_lines)
        
    write_jsonl(args.measure_log, logs)
