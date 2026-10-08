"""Leakage-safe feature preprocessing fitted on each holdout's training rows."""

import numpy as np


def fit_transform_train_test(X_train, X_test, transform_type):
    """Fit scaling statistics on training rows and apply them to both splits."""
    X_train = np.asarray(X_train, dtype=float)
    X_test = np.asarray(X_test, dtype=float)
    if transform_type == "none":
        return X_train.copy(), X_test.copy()
    if transform_type == "standardization":
        offset = np.mean(X_train, axis=0)
        scale = np.std(X_train, axis=0, ddof=0)
    elif transform_type == "minmax":
        offset = np.min(X_train, axis=0)
        scale = np.max(X_train, axis=0) - offset
    else:
        raise ValueError(f"Unknown transform type: {transform_type}")
    scale = np.asarray(scale, dtype=float)
    scale[scale == 0] = 1.0
    return (X_train - offset) / scale, (X_test - offset) / scale


def transform_binary_holdout(Atrain, Atest, Btrain, Btest, transform_type):
    """Scale a class-separated holdout using combined training statistics."""
    n_a, n_at = len(Atrain), len(Atest)
    train_t, test_t = fit_transform_train_test(
        np.vstack([Atrain, Btrain]), np.vstack([Atest, Btest]), transform_type)
    return train_t[:n_a], test_t[:n_at], train_t[n_a:], test_t[n_at:]
