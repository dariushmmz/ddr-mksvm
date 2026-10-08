"""Residual V3 with one train-only, data-driven deep-RBF bandwidth.

This is a separate opt-in variant.  It keeps Residual V3's architecture and
training schedule intact, but replaces the fixed ``alpha=1`` deep-RBF
bandwidth with the median non-zero pairwise distance of the initialized deep
training representation.  The scalar is computed once before optimization
and then remains frozen.
"""

from contextvars import ContextVar

import numpy as np
import torch

from ddr_mksvm.optim.alternating_trainer import AlternatingTrainer


_BANDWIDTH_RECORDS = ContextVar("adaptive_bandwidth_records", default=())


def clear_bandwidth_records():
    """Clear adaptive-bandwidth diagnostics in the current execution context."""
    _BANDWIDTH_RECORDS.set(())


def get_bandwidth_records():
    """Return copies of adaptive-bandwidth diagnostics for the current context."""
    return [dict(record) for record in _BANDWIDTH_RECORDS.get()]


def median_distance_bandwidth(Z):
    """Return the median non-zero Euclidean distance between feature rows."""
    Z = np.asarray(Z, dtype=np.float64)
    if Z.ndim != 2 or Z.shape[0] < 2:
        raise ValueError("Z must contain at least two row-wise observations")
    squared_norms = np.sum(Z * Z, axis=1)
    squared_distances = (
        squared_norms[:, None] + squared_norms[None, :] - 2.0 * (Z @ Z.T)
    )
    squared_distances = np.maximum(squared_distances, 0.0)
    upper = squared_distances[np.triu_indices(Z.shape[0], k=1)]
    scale = max(1.0, float(np.max(squared_norms)))
    positive = upper[upper > np.finfo(np.float64).eps * scale]
    if positive.size == 0:
        raise ValueError("cannot calibrate RBF bandwidth from a collapsed representation")
    return float(np.sqrt(np.median(positive)))


class AdaptiveBandwidthResidualTrainer(AlternatingTrainer):
    """Residual V3 with a frozen median-distance deep-RBF bandwidth."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if len(self.base_kernel_specs) != 1 or self.base_kernel_specs[0]["kind"] != "rbf":
            raise ValueError("adaptive bandwidth requires Residual V3's single deep RBF")
        self.adaptive_bandwidth = None

    def _calibrate_bandwidth(self, X_np):
        if self.adaptive_bandwidth is not None:
            return
        X_np = np.asarray(X_np, dtype=np.float64)
        was_training = self.f_theta.training
        self.f_theta.eval()
        try:
            with torch.no_grad():
                X_t = torch.tensor(X_np.T, dtype=torch.float32)
                Z = self.f_theta(X_t).detach().cpu().numpy()
        finally:
            self.f_theta.train(was_training)
        alpha = median_distance_bandwidth(Z)
        self.base_kernel_specs[0]["alpha"] = alpha
        self.adaptive_bandwidth = alpha
        record = {
            "adaptive_bandwidth": alpha,
            "median_rbf_similarity": float(np.exp(-0.5)),
            "n_training_samples": int(X_np.shape[1]),
        }
        _BANDWIDTH_RECORDS.set(_BANDWIDTH_RECORDS.get() + (record,))

    def fit(self, X_np, y):
        self._calibrate_bandwidth(X_np)
        return super().fit(X_np, y)

    def fit_one_vs_all(self, X_np, y_label, L):
        self._calibrate_bandwidth(X_np)
        return super().fit_one_vs_all(X_np, y_label, L)
