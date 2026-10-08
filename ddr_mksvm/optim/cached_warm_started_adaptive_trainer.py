"""Warm-started adaptive V3 with an opt-in invariant raw-Gram cache."""

from contextvars import ContextVar
import time

import numpy as np

from ddr_mksvm.optim.alternating_trainer import _numpy_gram
from ddr_mksvm.optim.warm_started_adaptive_trainer import (
    WarmStartedAdaptiveBandwidthTrainer,
)


_RAW_GRAM_CACHE_RECORDS = ContextVar("raw_gram_cache_records", default=())


def clear_raw_gram_cache_records():
    """Clear raw-anchor cache diagnostics in the current execution context."""
    _RAW_GRAM_CACHE_RECORDS.set(())


def get_raw_gram_cache_records():
    """Return copies of raw-anchor cache diagnostics in the current context."""
    return [dict(record) for record in _RAW_GRAM_CACHE_RECORDS.get()]


class CachedWarmStartedAdaptiveBandwidthTrainer(WarmStartedAdaptiveBandwidthTrainer):
    """Cache Residual V3's invariant raw training Gram once per fit."""

    def _prepare_raw_gram_cache(self, X_np):
        if self.raw_anchor_spec is None or "_cached_train_gram" in self.raw_anchor_spec:
            return
        uncached_spec = {
            key: value
            for key, value in self.raw_anchor_spec.items()
            if key != "_cached_train_gram"
        }
        started = time.perf_counter()
        cached = np.asarray(_numpy_gram(uncached_spec, X_np), dtype=np.float64)
        cached.setflags(write=False)
        self.raw_anchor_spec["_cached_train_gram"] = cached
        record = dict(
            n_training_samples=int(cached.shape[0]),
            cache_bytes=int(cached.nbytes),
            build_time_s=float(time.perf_counter() - started),
        )
        _RAW_GRAM_CACHE_RECORDS.set(_RAW_GRAM_CACHE_RECORDS.get() + (record,))

    def fit(self, X_np, y):
        self._prepare_raw_gram_cache(X_np)
        return super().fit(X_np, y)

    def fit_one_vs_all(self, X_np, y_label, L):
        self._prepare_raw_gram_cache(X_np)
        return super().fit_one_vs_all(X_np, y_label, L)
