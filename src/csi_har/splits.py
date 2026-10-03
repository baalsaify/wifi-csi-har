"""Cross-subject splits: no subject ever appears in both training and test data."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
from sklearn.model_selection import GroupKFold


def cross_subject_folds(
    subjects: np.ndarray, n_folds: int = 5, val_subjects: int = 3, seed: int = 42
) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Yield ``(train_idx, val_idx, test_idx)`` per fold.

    Test subjects come from GroupKFold. A few of the remaining subjects are held out
    as a validation set for early stopping, so the test fold is never used for tuning.
    """
    rng = np.random.default_rng(seed)
    indices = np.arange(len(subjects))
    for trainval_idx, test_idx in GroupKFold(n_splits=n_folds).split(indices, groups=subjects):
        pool = np.unique(subjects[trainval_idx])
        val_ids = rng.choice(pool, size=min(val_subjects, len(pool) - 1), replace=False)
        is_val = np.isin(subjects[trainval_idx], val_ids)
        yield trainval_idx[~is_val], trainval_idx[is_val], test_idx
