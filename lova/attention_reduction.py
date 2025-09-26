# File:         lova/attention_reduction.py
# Author:       Lea Button
# Date:         25-09-2025
# Description:  Reduces attention matrices to line-level summaries.

import torch
from typing import List, Dict, Tuple

class AttentionReducer:
    """ Reduces raw attention tensors into line-level attention summaries per layer. """
    def __init__(self):
        pass


    def build_token_to_line_map(self, line_map: Dict[int, str], offset_mapping: List[Tuple[int, int]]) -> List[int]:
        """
        Build mapping from token index to line number.

        Parameters
        ----------
        line_map : Dict[int, str]
            Mapping from line number to line content.
        offset_mapping : List[Tuple[int, int]]
            List of (start_offset, end_offset) for each token.
            
        Returns
        -------
        token_to_line : List[int]
            List where index is token index and value is corresponding line number.
        """
        code_lines = list(line_map.items())  # [(line_no, line_str)]
        line_spans = []
        total_offset = 0
        for line_no, line in code_lines:
            line_header = f"[{line_no}] "
            line_len = len(line_header + line) + 1  # +1 for \n
            span = (line_no, total_offset, total_offset + line_len)
            line_spans.append(span)
            total_offset += line_len

        token_to_line = []
        for start, end in offset_mapping:
            matched_line = None
            for line_no, span_start, span_end in line_spans:
                if start >= span_start and start < span_end:
                    matched_line = line_no
                    break
            token_to_line.append(matched_line)

        return token_to_line


    def reduce_attention(self, attention: torch.Tensor, token_to_line: List[int]) -> Tuple[torch.Tensor, List[int]]:
        """
        Reduce full attention map to (num_lines, num_layers).
        
        Parameters
        ----------
        attention : torch.Tensor
            Attention tensor of shape (num_layers, seq_len, seq_len).
        token_to_line : List[int]
            Mapping from token index to line number.
            
        Returns
        -------
        attn_tensor : torch.Tensor
            Reduced attention tensor of shape (num_lines, num_layers).
        sorted_lines : List[int]
            List of valid line numbers corresponding to rows in the reduced attention tensor.
        """
        last_token_attention = attention[:, -1, :]  # (L, S)
        line_attn = {}
        for token_idx, line_no in enumerate(token_to_line):
            if line_no is not None:
                if line_no not in line_attn:
                    line_attn[line_no] = torch.zeros(attention.shape[0], device=attention.device)
                line_attn[line_no] += last_token_attention[:, token_idx]

        sorted_lines = sorted(line_attn.keys())
        attn_tensor = torch.stack([line_attn[ln] for ln in sorted_lines])  # (num_valid_lines, num_layers)

        return attn_tensor, sorted_lines
   

    def align_and_diff(self, base_attn: torch.Tensor, highlighted_attn: torch.Tensor,
                       valid_lines: List[int], h_lines: List[int]) -> torch.Tensor:
        """
        Align a highlighted attention matrix to the base attention by line numbers and compute the difference.

        Parameters
        ----------
        base_attn : torch.Tensor
            Base attention tensor of shape (num_lines, num_layers).
        highlighted_attn : torch.Tensor
            Highlighted attention tensor of shape (num_highlighted_lines, num_layers).
        valid_lines : List[int]
            List of line numbers corresponding to rows in base_attn.
        h_lines : List[int]
            List of line numbers corresponding to rows in highlighted_attn.
            
        Returns
        -------
        torch.Tensor
            Difference tensor of shape (num_lines, num_layers).
        """
        aligned_h_attn = []
        for line in valid_lines:
            if line in h_lines:
                idx = h_lines.index(line)
                aligned_h_attn.append(highlighted_attn[idx])
            else:
                aligned_h_attn.append(torch.zeros_like(base_attn[0]))
        aligned_h_attn = torch.stack(aligned_h_attn)  # (num_lines, num_layers)

        return aligned_h_attn - base_attn
