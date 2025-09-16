# --------------------------------------------------------
# WoundAmbit
# Copyright (c) 2025 Vanessa Borst, Timo Dittus and Contributors.
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import contextlib
import os
import random
from typing import List

import numpy as np
import torch
from beartype import beartype


def get_splits(indices: np.ndarray, train_ratio: float, val_ratio: float, seed: int = None) -> (
        np.ndarray, np.ndarray, np.ndarray):
    """
    Splits the given indices into train, validation and test sets. Randomizes the order if a seed is given.
    param indices: The indices to split.
    param train_ratio: The ratio of the indices to use for training.
    param val_ratio: The ratio of the indices to use for validation.
    param seed: The seed to use for the random number generator.
    return: The train, validation and test indices.
    """
    assert train_ratio + val_ratio <= 1.0
    if seed is not None:
        with set_np_temp_seed(seed):
            np.random.shuffle(indices)
    n = len(indices)
    val_start = int(train_ratio * n)
    test_start = val_start + int(val_ratio * n)
    train_indices = indices[:val_start]
    val_indices = indices[val_start:test_start]
    test_indices = indices[test_start:]
    assert n == len(train_indices) + len(val_indices) + len(test_indices)
    # assert that there are no overlaps
    assert len(set(train_indices).intersection(set(val_indices))) == 0
    assert len(set(train_indices).intersection(set(test_indices))) == 0
    assert len(set(val_indices).intersection(set(test_indices))) == 0
    return train_indices, val_indices, test_indices


@beartype
def get_folds(indices: List[int], k: int, seed: int = None) -> List[List[int]]:
    """
    Splits the given indices into k folds for k-fold cross-validation. Randomizes the order if a seed is given.
    param indices: The indices to split.
    param k: The number of folds.
    param seed: The seed to use for the random number generator.
    return: The k-fold indices.
    """
    assert indices is not None and len(indices) > 0, "Indices must not be empty"
    np_indices = np.array(indices)
    if seed is not None:
        with set_np_temp_seed(seed):
            np.random.shuffle(np_indices)
    all_indices: List = np_indices.tolist()
    n = len(all_indices)
    fold_size = n // k
    folds = []

    for i in range(k):
        start = i * fold_size
        end = (i + 1) * fold_size if i < k - 1 else n
        fold_indices = all_indices[start:end]
        folds.append(fold_indices)

    return folds


@contextlib.contextmanager
def set_np_temp_seed(seed: int):
    """
    Sets the numpy random seed for the duration of the context.
    param seed: The seed to use.
    """
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def ensure_reproducibility(seed: int):
    # https://pytorch.org/docs/stable/notes/randomness.html
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)


def use_deterministic():
    # https://pytorch.org/docs/stable/notes/randomness.html
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    ensure_reproducibility(worker_seed)
