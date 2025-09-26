# File:         data/bigvul_loader.py
# Author:       Lea Button
# Date:         25-09-2025
# Description:  Loader for the BigVul dataset, which processes and caches 
#               examples of vulnerable code. 

import pandas as pd
import os
import pickle
import difflib
import torch

class BigVulDataset:
    """ Dataset class for the BigVul dataset. """
    def __init__(self, csv_path: str, limit: int = None, 
                 load_attention: bool = False, model_name: str = None):
        """
        Initializes the BigVulDataset and caches the processed examples.
        
        Parameters
        ----------
        csv_path : str
            Path to the CSV file containing the dataset.
        limit : int, optional
            Limit the number of examples to load. Default is None (load all).
        load_attention : bool, optional
            Whether to load precomputed attention matrices. Default is False.
        model_name : str, optional
            Name of the model for which attention matrices were computed.
        """
        self.cache_path = csv_path.replace(".csv", "_cache.pkl")
        self.load_attention = load_attention
        self.model_name = model_name

        if os.path.exists(self.cache_path):
            with open(self.cache_path, 'rb') as f:
                self.data = pickle.load(f)
                if limit and len(self.data) > limit:
                    self.data = self.data[:limit]
            return
        
        self.df = pd.read_csv(csv_path)
        self.df.dropna(subset=["func_before", "func_after"], inplace=True)
        # Filter for vulnerable functions
        self.df = self.df[self.df["vul"] == 1]

        self.data = self._process_examples()
        
        if limit and len(self.data) > limit:
            self.data = self.data[:limit]

        with open(self.cache_path, 'wb') as f:
            pickle.dump(self.data, f)

    def _process_examples(self):
        """
        Processes the DataFrame to create a list of examples.

        Returns
        -------
        examples : List[Dict]
            List of processed examples with numbered code and vulnerability lines.
        """
        examples = []
        count = 0
        
        for i, row in self.df.iterrows():
            func_before = row["func_before"].strip().split('\n')
            func_after = row["func_after"].strip().split('\n')
            commit_id = row["commit_id"]
            
            diff = list(difflib.ndiff(func_before, func_after))
            
            vuln_lines = []
            line_idx = 1
            
            for line in diff:
                if line.startswith('- '):
                    vuln_lines.append(line_idx)
                    line_idx += 1
                elif line.startswith('+ '):
                    continue
                else:
                    line_idx += 1
            
            code_raw = '\n'.join(func_before)
            code_numbered, line_map = self._number_lines(code_raw)
            

            if len(code_raw) > 2500:
                count += 1
                continue
        
            examples.append({
                "id": f"{commit_id}_{i}",
                "commit_id": commit_id,
                "code_raw": code_raw,
                "code_numbered": code_numbered,
                "line_map": line_map,
                "vuln_lines": vuln_lines,
            })
            
        print(f"Skipped {count} examples due to code length > 2500 characters.")
        return examples 

    def _number_lines(self, code: str):
        """ Numbers the lines of code and creates a mapping from line numbers to code lines. """
        lines = code.strip().split('\n')
        numbered = [f"[{i+1}] {line}" for i, line in enumerate(lines)]
        line_map = {i+1: line for i, line in enumerate(lines)}
        return '\n'.join(numbered), line_map

    def __len__(self):
        """ Returns the number of examples in the dataset. """
        return len(self.data)

    def __getitem__(self, idx):
        """ Retrieves an example by index, optionally loading attention data. """
        ex = self.data[idx]

        if self.load_attention and self.model_name:
                attn_path = os.path.join(
                    "cache/attn_matrices",
                    self.model_name.replace("/", "_"),
                    f"{ex['id']}.pt"
                )
                if os.path.exists(attn_path):
                    ex[f"attn_matrix-{self.model_name}"] = torch.load(
                        attn_path)
                else:
                    ex[f"attn_matrix-{self.model_name}"] = None 

        return ex
