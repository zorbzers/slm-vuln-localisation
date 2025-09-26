# File:         data/attn_dataset.py
# Author:       Lea Button
# Date:         25-09-2025
# Description:  Dataset wrapper that lazily loads attention matrices from cache at runtime.

import os
import json
import torch
from torch.utils.data import Dataset

class AttnDataset(Dataset):
    """ Dataset that loads attention matrices from cached .pt files on-the-fly. """
    def __init__(self, json_path: str, model_name: str, cache_dir: str, shots: int = 0, device="cpu"):
        """
        Initializes the AttnDataset.
        
        Parameters
        ----------
        json_path : str
            Path to the JSON file containing dataset metadata.
        model_name : str
            Name of the model for which attention matrices were computed.
        cache_dir : str
            Directory where cached attention .pt files are stored.
        shots : int, optional
            Number of few-shot examples used (affects cache path). Default is 0.
        device : str, optional
            Device to load tensors onto ("cpu" or "cuda"). Default is "cpu".
        """
        self.json_path = json_path
        self.model_id = model_name.replace("/", "_")
        self.cache_dir = cache_dir
        self.shots = shots
        self.device = device

        # Load metadata only
        with open(json_path) as f:
            self.metadata = json.load(f)

    def __len__(self):
        """ Returns the number of examples in the dataset. """
        return len(self.metadata)

    def __getitem__(self, idx):
        """ Retrieves an example by index, loading its attention matrix from cache. """
        ex = self.metadata[idx]
        ex_id = ex["id"]

        # Path to cached attention
        pt_path = os.path.join(
            self.cache_dir,
            self.model_id,
            f"{self.shots}_shot",
            f"{ex_id}.pt"
        )
        if not os.path.exists(pt_path):
            raise FileNotFoundError(f"Missing attention cache for {ex_id}: {pt_path}")

        # Load attention from cache
        pt_data = torch.load(pt_path, map_location=self.device)

        if isinstance(pt_data, dict):
            attn_matrix = pt_data["attn_matrix"]
            valid_lines = pt_data["valid_lines"]
        elif isinstance(pt_data, torch.Tensor):
            attn_matrix = pt_data
            valid_lines = list(range(attn_matrix.shape[0]))
        else:
            raise TypeError(f"Unexpected cache format for {ex_id}: {type(pt_data)}")

        # Reduce if needed
        if attn_matrix.dim() == 3:
            # (num_lines, num_lines, num_layers) → (num_lines, num_layers)
            X = attn_matrix.mean(dim=0)
        elif attn_matrix.dim() == 2:
            X = attn_matrix
        else:
            raise ValueError(f"Unexpected tensor shape {attn_matrix.shape} for {ex_id}")

        # Build labels
        y = torch.zeros(len(valid_lines), device=self.device)
        for i, orig_line in enumerate(valid_lines):
            if orig_line in ex["vuln_lines"]:
                y[i] = 1

        return X.to(self.device), y.to(self.device), ex
