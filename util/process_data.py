"""
Simple data processing utilities: normalization.
All functions return transformed data and a state dictionary for reverse operations.
"""

import numpy as np


def norm_normal(data):
    """
    Normalize data to zero mean and unit variance per feature.

    Args:
        data: (n_samples, n_features)

    Returns:
        data_norm: normalized data
        state: dict with 'mean' and 'std'
    """
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    std[std == 0] = 1.0  # avoid division by zero
    data_norm = (data - mean) / std
    state = {'mean': mean, 'std': std}
    return data_norm, state


def reverse_norm_normal(data_norm, state):
    """Reverse normalization."""
    return data_norm * state['std'] + state['mean']