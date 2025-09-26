# File:         scripts/utils.py
# Author:       Lea Button
# Date:         25-09-2025
# Description:  Utility functions for measuring resource usage and logging.

import os, time, json, psutil
from contextlib import contextmanager
import torch
from torch.nn.utils.rnn import pad_sequence


def _gpu_reset():
    """Reset GPU peak memory stats."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()


def _gpu_peak_bytes():
    """Get peak GPU memory usage in bytes since last reset."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        return int(torch.cuda.max_memory_allocated())
    return 0


@contextmanager
def measured_phase(phase_name, log_sink, extra=None):
    """Context manager to time a code block and record RAM/VRAM peaks."""
    proc = psutil.Process(os.getpid())
    start_ram = proc.memory_info().rss
    _gpu_reset()
    t0 = time.perf_counter()
    try:
        yield
    finally:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        wall = time.perf_counter() - t0
        end_ram = proc.memory_info().rss
        entry = {
            "phase": phase_name,
            "wall_clock_s": wall,
            "delta_ram_bytes": max(0, end_ram - start_ram),
            "peak_vram_bytes": _gpu_peak_bytes(),
            "pid": os.getpid(),
            "hostname": os.uname().nodename if hasattr(os, "uname") else ""
        }
        if extra:
            entry.update(extra)
        if isinstance(log_sink, list):
            log_sink.append(entry)
        elif isinstance(log_sink, str):
            os.makedirs(os.path.dirname(log_sink), exist_ok=True)
            with open(log_sink, "a") as f:
                f.write(json.dumps(entry) + "\n")


def write_jsonl(path, rows):
    """Append rows (list of dicts) to a JSONL file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    
def collate_fn(batch):
    """
    Collate function for variable-length line representations.
    Pads to the maximum number of lines in the batch.
    
    Parameters
    ----------
    batch : list of tuples
        Each tuple is (X, y, metadata) where:

    Returns
    ----------
    padded_X : Tensor
        Padded line representations (B, max_len, num_layers)
    padded_y : Tensor
        Padded labels (B, max_len)
    mask : Tensor
        Boolean mask for valid (unpadded) positions (B, max_len)
    metas : list
        List of metadata dicts for each example   
    """
    Xs, ys, metas = zip(*batch)

    lengths = [x.size(0) for x in Xs]

    padded_X = pad_sequence(Xs, batch_first=True)  # (B, max_len, num_layers)
    padded_y = pad_sequence(ys, batch_first=True)  # (B, max_len)

    # Build mask: True for valid (unpadded) positions
    max_len = padded_X.size(1)
    mask = torch.zeros(len(Xs), max_len, dtype=torch.bool)
    for i, l in enumerate(lengths):
        mask[i, :l] = 1

    return padded_X, padded_y, mask, metas