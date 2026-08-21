"""
Validates that ddr_mksvm.config_adapter.kernel_spec_from_config, combined
with ddr_mksvm's own kernel classes, reproduces EXACTLY the Gram matrix
that unit_of_work_deterministic_binary.py / unit_of_work_deterministic_multiclass.py's
own _compute_kernel would produce, for every kernel string that appears
in either DATASET_CONFIG.

This is the numerical guarantee behind unit_of_work_ddr_binary.py's
docstring claim that dnn_on=False, mkl_on=False, dro_on=False reproduces
the deterministic baseline exactly (Proposition 4.8) for ANY dataset in
the registry, not just one hand-checked case.

Requires cvxpy (unit_of_work_deterministic_binary.py imports it at module
level even though these particular functions don't use it) -- skipped
automatically if unavailable.
"""

import pytest

cp = pytest.importorskip("cvxpy")
import numpy as np

from unit_of_work_deterministic_binary import (
    DATASET_CONFIG as BINARY_CONFIG,
    _parse_kernel as _parse_kernel_binary,
    _compute_kernel as _compute_kernel_binary,
)
from unit_of_work_deterministic_multiclass import (
    DATASET_CONFIG as MULTICLASS_CONFIG,
    _parse_kernel as _parse_kernel_multiclass,
    _compute_kernel as _compute_kernel_multiclass,
)
from ddr_mksvm.config_adapter import kernel_spec_from_config
from ddr_mksvm.kernels.base_kernels import build_kernel


def _check_all_kernels(dataset_config, parse_fn, compute_fn):
    rng = np.random.default_rng(11)
    dati = rng.normal(size=(6, 20))  # n=6 features, m=20 points

    kernel_strs = {cfg["kernel"] for cfg in dataset_config.values()}
    assert kernel_strs, "expected at least one kernel string in the registry"

    for kernel_str in kernel_strs:
        specs, K_ref, param = kernel_spec_from_config(kernel_str, dati, parse_fn, compute_fn)
        spec = specs[0]
        kernel_obj = build_kernel(spec)
        kernel_obj.fit(dati)
        K_new = kernel_obj.gram(dati)
        np.testing.assert_allclose(
            K_new, K_ref, rtol=1e-8, atol=1e-8,
            err_msg=f"kernel_spec_from_config mismatch for kernel_str={kernel_str!r}"
        )


def test_binary_dataset_kernels_match():
    _check_all_kernels(BINARY_CONFIG, _parse_kernel_binary, _compute_kernel_binary)


def test_multiclass_dataset_kernels_match():
    _check_all_kernels(MULTICLASS_CONFIG, _parse_kernel_multiclass, _compute_kernel_multiclass)
