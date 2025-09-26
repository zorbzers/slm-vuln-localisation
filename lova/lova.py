# File:         lova/lova.py
# Author:       Lea Button
# Date:         07-07-2025
# Description:  Main LoVA class combining attention extraction, reduction, and classification.

import time
import torch
from typing import List, Dict
import tqdm
import numpy as np
import joblib
from sklearn.metrics import precision_score, recall_score, f1_score

from lova.attention_extraction import AttentionExtractor
from lova.attention_reduction import AttentionReducer
from lova.classifier_configs import CLASSIFIER_CONFIGS
from prompts.prompt_builder import PromptBuilder

class Lova(AttentionExtractor, AttentionReducer):
    """
    Main LoVA class combining attention extraction, reduction, and classification.
    """
    def __init__(self, model=None, tokenizer=None, 
                 device = 'cuda' if torch.cuda.is_available() else 'cpu', 
                 shots=0, classifier_name="BiLSTM_moderate"):
        """
        Initializes the LoVA system with model, tokenizer, device, shots, and classifier configuration.
        
        Parameters
        ----------
        model : PreTrainedModel, optional
            Pre-trained model for attention extraction.
        tokenizer : PreTrainedTokenizer, optional
            Pre-trained tokenizer for attention extraction.
        device : str, optional
            Device to run the model on ('cpu' or 'cuda'). Default is 'cuda' if available.
        shots : int, optional
            Number of example shots to include in prompts. Default is 0.
        classifier_name : str, optional
            Name of the classifier configuration to use. Default is "BiLSTM_moderate".
        """
        if model is not None and tokenizer is not None:
            AttentionExtractor.__init__(self, model, tokenizer, device)
        AttentionReducer.__init__(self)
        self.device = device
        self.shots = shots
        self.prompt_builder = PromptBuilder(shots=shots)
        
        self.classifier_name = classifier_name
        self.classifier = None
        self.optim_class = None
        self.optim_kwargs = None
        self.best_threshold = 0.5 
    

    def initialize_classifier(self, input_dim: int):
        """ Initializes the classifier model based on the specified configuration. """
        constructor = CLASSIFIER_CONFIGS[self.classifier_name]
        model, optim_class, optim_kwargs = constructor(input_dim)
        self.classifier = model.to(self.device) if hasattr(model, "to") else model
        self.optim_class = optim_class
        self.optim_kwargs = optim_kwargs
        
    
    def get_reduced_attention(self, prompt: str, line_map: Dict[int, str]):
        """ Extracts and reduces attention for a given prompt and line map. """
        raw_attn, offsets = self.extract_raw_attention(prompt)
        token_to_line = self.build_token_to_line_map(line_map, offsets)
        return self.reduce_attention(raw_attn, token_to_line)
       
        
    def extract_lova_matrix(self, code: str, line_map: Dict[int, str]) -> torch.Tensor:
        """ Extracts the LoVA attention matrix for a given code snippet and line map. """
        base_prompt = self.prompt_builder.build_base_prompt(code)
        base_attn, valid_lines = self.get_reduced_attention(base_prompt, line_map)

        highlighted_prompts = self.prompt_builder.build_all_highlighted_prompts(code, valid_lines)
        lova_matrices = []

        print(f"DEBUG: len(highlighted_prompts) = {len(highlighted_prompts)}")

        for h_prompt in tqdm.tqdm(highlighted_prompts, desc="LOVA attention"):
            h_attn, h_lines = self.get_reduced_attention(h_prompt, line_map)
            diff = self.align_and_diff(base_attn, h_attn, valid_lines, h_lines)
            lova_matrices.append(diff.unsqueeze(0))

        return torch.cat(lova_matrices, dim=0), valid_lines 
    
    
    def flatten_lova_matrix(self, mat: torch.Tensor) -> torch.Tensor:
        """ Flattens the LoVA matrix by averaging attention scores across heads. """
        return mat.mean(dim=1)  
    

    def train_classifier(self, train_loader, epochs=5, pos_weight=None):
        """ Trains the classifier using the provided DataLoader.
        
        Parameters
        ----------
        train_loader : DataLoader
            DataLoader providing training data batches.
        epochs : int, optional
            Number of training epochs. Default is 5.
        pos_weight : float, optional
            Weight for positive class in loss function to handle class imbalance. Default is None.
            
        Yields
        ------
        dict
            Dictionary containing epoch number and training loss.
        """
        if self.classifier is None:
            raise RuntimeError("Classifier not initialized. Call initialize_classifier first.")

        if isinstance(self.classifier, torch.nn.Module):
            loss_fn = torch.nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor([pos_weight], device=self.device)
            ) if pos_weight else torch.nn.BCEWithLogitsLoss()

            optimizer = torch.optim.Adam(self.classifier.parameters(), **self.optim_kwargs)

            self.classifier.train()
            for epoch in range(epochs):
                total_loss = 0
                for X, y, mask, _ in train_loader:
                    X, y, mask = X.to(self.device), y.to(self.device), mask.to(self.device)

                    logits = self.classifier(X)  # (B, max_len)
                    logits = logits[mask]
                    y = y[mask]

                    loss = loss_fn(logits, y)
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    total_loss += loss.item()
                yield {"epoch": epoch + 1, "train_loss": total_loss}
        else:
            all_X, all_y = [], []
            for X, y, mask, _ in train_loader:
                all_X.append(X[mask].cpu().numpy())
                all_y.append(y[mask].cpu().numpy())
            all_X = np.vstack(all_X)
            all_y = np.concatenate(all_y)
            self.classifier.fit(all_X, all_y)
            yield {"epoch": 1, "train_loss": None}


    def tune_threshold(self, dataloader):
        """
        Find the threshold that maximises F1 on validation data.
        Works with padded batches from AttnDataset + collate_fn.
        
        Parameters
        ----------
        dataloader : DataLoader
            DataLoader providing validation data batches.
            
        Returns
        -------
        tuple
            Best threshold and corresponding F1 score.
        """
        if not isinstance(self.classifier, torch.nn.Module):
            return 0.5, 0.0  

        preds_all, labels_all = [], []
        self.classifier.eval()
        with torch.no_grad():
            for X, y, mask, _ in dataloader:
                X, y, mask = X.to(self.device), y.to(self.device), mask.to(self.device)
                logits = self.classifier(X)
                probs = torch.sigmoid(logits)
                preds_all.extend(probs[mask].cpu().numpy())
                labels_all.extend(y[mask].cpu().numpy())

        preds_all = np.array(preds_all)
        labels_all = np.array(labels_all)

        best_thresh, best_f1 = 0.5, 0.0
        for thresh in np.linspace(0.1, 0.9, 17):
            preds = (preds_all > thresh).astype(int)
            f1 = f1_score(labels_all, preds, zero_division=0)
            if f1 > best_f1:
                best_f1, best_thresh = f1, thresh

        self.best_threshold = best_thresh  
        return best_thresh, best_f1

    
    def evaluate(self, val_loader):
        """
        Evaluate the classifier on validation data.
        
        Parameters
        ----------
        val_loader : DataLoader
            DataLoader providing validation data batches.
        
        Returns
        -------
        dict
            Dictionary containing precision, recall, F1 score, and exact match percentage.
        """
        preds_all, labels_all = [], []
        em_total, n_examples = 0, 0

        if isinstance(self.classifier, torch.nn.Module):
            self.classifier.eval()
            with torch.no_grad():
                for X, y, mask, _ in val_loader:
                    X, y, mask = X.to(self.device), y.to(self.device), mask.to(self.device)

                    logits = self.classifier(X)
                    probs = torch.sigmoid(logits)
                    preds = (probs > self.best_threshold).int()

                    # Flatten for precision/recall/F1
                    preds_all.extend(preds[mask].cpu().tolist())
                    labels_all.extend(y[mask].cpu().tolist())

                    # Per-example exact match
                    for i in range(X.size(0)):
                        true_lines = y[i][mask[i]].cpu().tolist()
                        pred_lines = preds[i][mask[i]].cpu().tolist()
                        if true_lines == pred_lines:
                            em_total += 1
                        n_examples += 1
        else:
            preds_all, labels_all = [], []
            em_total, n_examples = 0, 0

            for X, y, mask, _ in val_loader:
                # Extract only the masked rows for this batch
                X_masked = X[mask].cpu().numpy()
                y_masked = y[mask].cpu().numpy()

                # Predict for just this batch
                preds_masked = self.classifier.predict(X_masked)

                # Flatten for P/R/F1
                preds_all.extend(preds_masked.tolist())
                labels_all.extend(y_masked.tolist())

                # --- Grouped by example for EM ---
                for i in range(len(y)):
                    true_lines = y[i][mask[i]].cpu().tolist()
                    pred_lines = preds_masked[: len(true_lines)].tolist()
                    preds_masked = preds_masked[len(true_lines):]

                    if true_lines == pred_lines:
                        em_total += 1
                    n_examples += 1
                    
        return {
            "precision": precision_score(labels_all, preds_all, zero_division=0) * 100,
            "recall": recall_score(labels_all, preds_all, zero_division=0) * 100,
            "f1": f1_score(labels_all, preds_all, zero_division=0) * 100,
            "exact_match": (em_total / n_examples) * 100 if n_examples > 0 else 0.0,
        }


    def save_classifier(self, path: str):
        """Save the trained classifier (torch or sklearn/XGB)."""
        meta = {"threshold": self.best_threshold}
        if isinstance(self.classifier, torch.nn.Module):
            torch.save(
                {"state_dict": self.classifier.state_dict(), "meta": meta},
                path
            )
        else:
            joblib.dump((self.classifier, meta), path)


    def load_classifier(self, path: str, input_dim: int):
        """Load a saved classifier."""
        if path.endswith(".pt"):
            ckpt = torch.load(path, map_location=self.device, weights_only=False)
            self.initialize_classifier(input_dim=input_dim)
            self.classifier.load_state_dict(ckpt["state_dict"])
            self.best_threshold = ckpt.get("meta", {}).get("threshold", 0.5)
        else:
            self.classifier, meta = joblib.load(path)
            self.best_threshold = meta.get("threshold", 0.5)
