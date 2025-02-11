"""
Helper functions for general purposes in the sus-adv-gen project.
This module provides utility functions for GPU memory management, directory handling,
and code execution timing.
Functions:
    empty_gpu_cache(): Clears GPU memory cache based on available backend (MPS or CUDA).
    ensure_directory_exists(directory_path): Creates directory if it doesn't exist.
    timer_context(name): Context manager for measuring execution time of code blocks.
"""

import os
import torch


def empty_gpu_cache() -> None:
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()


def ensure_directory_exists(directory_path: str) -> None:
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory created: {directory_path}")
    else:
        print(f"Directory already exists: {directory_path}")
