"""
Bridges the dataset-config kernel-string convention used in the
corrected, fully-parameterized unit_of_work_deterministic_binary.py /
unit_of_work_deterministic_multiclass.py ("hom_linear", "inhom_quadratic",
"inhom_cubic", "gaussian_rbf", ...) to ddr_mksvm's own base_kernel_specs
convention (kind='poly'|'rbf', degree, c/alpha).

Deliberately reuses the deterministic scripts' OWN _parse_kernel /
_compute_kernel functions (imported at the call site, not reimplemented
here) so that DDR-MKSVM's legacy path (eta one-hot on this single spec,
dnn_on=False, mkl_on=False, dro_on=False) is guaranteed to use exactly
the same Gram matrix as the deterministic baseline -- there is no room
for the two to silently drift apart as DATASET_CONFIG evolves.
See tests/test_config_adapter.py for the numerical cross-check.
"""


def kernel_spec_from_config(kernel_str, dati, parse_kernel_fn, compute_kernel_fn):
    """
    Parameters
    ----------
    kernel_str        : e.g. "inhom_quadratic", "hom_linear", "gaussian_rbf"
                         (a DATASET_CONFIG[...]["kernel"] value)
    dati               : (n, m) ndarray, features as columns
    parse_kernel_fn    : the deterministic script's own _parse_kernel
    compute_kernel_fn  : the deterministic script's own _compute_kernel

    Returns
    -------
    base_kernel_specs : list[dict] -- ddr_mksvm kernel spec, length 1
    K_legacy           : (m, m) ndarray -- exactly what the deterministic
                          baseline would compute for this kernel_str
    kernel_param        : the c (polynomial) or alpha (RBF) value used,
                          needed later to build compatible test-time
                          kernels via the deterministic script's own
                          _compute_kernel_test
    """
    if kernel_str == "gaussian_rbf":
        K, alpha = compute_kernel_fn(dati, "gaussian_rbf", None, None)
        spec = dict(kind="rbf", alpha=float(alpha))
        return [spec], K, alpha
    else:
        c_type, d = parse_kernel_fn(kernel_str)
        K, c_val = compute_kernel_fn(dati, "polynomial", c_type, d)
        spec = dict(kind="poly", degree=d, c=float(c_val))
        return [spec], K, c_val
