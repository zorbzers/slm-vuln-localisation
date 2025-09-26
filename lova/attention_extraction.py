# File:         lova/attention_extraction.py
# Author:       Lea Button
# Date:         25-09-2025
# Description:  Extracts raw attention tensors from a pre-trained model.

import torch
from transformers import PreTrainedTokenizer, PreTrainedModel

class AttentionExtractor:
    """ Extracts raw attention tensors from a pre-trained model. """
    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, device: str = "cuda"):
        """
        Initializes the AttentionExtractor with a model and tokenizer.
        
        Parameters
        ----------
        model : PreTrainedModel
            The pre-trained model from which to extract attention.
        tokenizer : PreTrainedTokenizer
            The tokenizer corresponding to the model.
        device : str, optional
            Device to run the model on ("cuda" or "cpu"). Default is "cuda".
        """
        self.model = model.to(device)
        self.model.eval()
        self.tokenizer = tokenizer
        self.device = device


    def extract_raw_attention(self, prompt: str) -> torch.Tensor:
        """
        Extracts raw attention tensors from the model for a given prompt.
        
        Parameters
        ----------
        prompt : str
            The input text prompt.
            
        Returns
        -------
        attn_combined : torch.Tensor
            Attention tensor of shape (num_layers, seq_len, seq_len).
        offset_mapping : List[Tuple[int, int]]
            List of (start_offset, end_offset) for each token in the input.
        """
        with torch.no_grad():
            inputs = self.tokenizer(prompt, return_offsets_mapping=True, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            output = self.model(**inputs, output_attentions=True)
            attn = torch.stack(output.attentions)  # shape: (L, B, H, S, S)
        
        L, B, H, S, _ = attn.shape
        attn = attn.squeeze(1)           # (L, H, S, S)
        attn_combined = attn.sum(dim=1)  # (L, S, S)

        offset_mapping = inputs["offset_mapping"][0].cpu().tolist()

        return attn_combined, offset_mapping