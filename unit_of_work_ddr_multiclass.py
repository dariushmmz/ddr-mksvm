"""
unit_of_work_ddr_multiclass.py

DDR-MKSVM one-vs-all multiclass unit of work. Previously flagged in this
project's README as an unported gap -- this closes it, aligned with the
corrected unit_of_work_deterministic_multiclass.py (imports its own
DATASET_CONFIG, _apply_transform, _parse_kernel, _compute_kernel,
_compute_kernel_test directly, same reasoning as unit_of_work_ddr_binary.py).

Note DATASET_CONFIG here now correctly includes heart_disease (5 classes)
and dermatology (6 classes) as multiclass datasets -- matching what the
earlier reproduction-report review flagged as the likely-correct reading
of the base paper's Table 3/5 formatting.

Design choice: the deep feature extractor f_theta and kernel mixture eta
are SHARED across all L one-vs-all binary subproblems (a single learned
representation, jointly optimized against the mean one-vs-all hinge loss
across classes -- see ddr_mksvm/optim/alternating_trainer.py's
fit_one_vs_all), rather than training L independent representations.
This mirrors how the deterministic script already shares one kernel/Gram
matrix K across all L subproblems; DDR-MKSVM shares the entire learned
feature map the same way.
"""

import numpy as np

from holdouts_train_test_multiclass import holdouts_train_test_multiclass
from unit_of_work_deterministic_multiclass import (
    DATASET_CONFIG,
    _apply_transform,
    _parse_kernel,
    _compute_kernel,
    _compute_kernel_test,
)
from ddr_mksvm.optim.convex_subproblem import train_with_nu_search
from ddr_mksvm.config_adapter import kernel_spec_from_config


def unit_of_work_ddr_multiclass(DATA, dataset_name="iris", dnn_on=True, mkl_on=True,
                                 dro_on=True, epsilon=0.001, extra_kernel_specs=None,
                                 n_outer=6, n_inner=15, seed=None):
    """
    seed : int or None -- see unit_of_work_ddr_binary.py's docstring for
        the same semantics (drives both the holdout split and the deep
        feature extractor's initialization when set).
    """
    if dataset_name not in DATASET_CONFIG:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(DATASET_CONFIG.keys())}")

    config = DATASET_CONFIG[dataset_name]
    transform_type = config["data_transform"]
    kernel_str = config["kernel"]
    expected_classes = config["classes"]

    # ------------------------------------------------------------------
    # Transform + CLASS-column normalization + holdout split
    # ------------------------------------------------------------------
    DATA_transformed = _apply_transform(DATA, transform_type)
    if DATA_transformed.columns[-1] != "CLASS":
        cols = list(DATA_transformed.columns)
        cols[-1] = "CLASS"
        DATA_transformed.columns = cols

    testingsamplesize = 0.25
    DATAtrain, DATAtest = holdouts_train_test_multiclass(DATA_transformed, testingsamplesize,
                                                          random_state=seed)

    y_label = DATAtrain[:, -1].astype(int)
    dati = DATAtrain[:, :-1].T
    n, m_train_tot = dati.shape

    Xtest = DATAtest[:, :-1].T
    y_test_all = DATAtest[:, -1].astype(int)
    m_test_tot = Xtest.shape[1]

    L = int(max(y_label.max(), y_test_all.max()))
    if L != expected_classes:
        print(f"  WARNING: Expected {expected_classes} classes, found {L} in data.")

    # ------------------------------------------------------------------
    # Kernel setup
    # ------------------------------------------------------------------
    base_kernel_specs, K_legacy, kernel_param = kernel_spec_from_config(
        kernel_str, dati, _parse_kernel, _compute_kernel)

    if mkl_on and extra_kernel_specs:
        base_kernel_specs = base_kernel_specs + list(extra_kernel_specs)
    else:
        mkl_on = False

    is_pure_legacy = not dnn_on and not mkl_on and not dro_on
    trainer = None

    u_vect = np.zeros((m_train_tot, L))
    b_vect = np.zeros(L)
    Du_vect = np.zeros((m_train_tot, L))

    # ------------------------------------------------------------------
    # TRAINING PHASE (one-vs-all, L binary sub-problems)
    # ------------------------------------------------------------------
    if not dnn_on and not mkl_on:
        # Legacy (Proposition 4.8) or DRO-only: single fixed kernel, no
        # learned feature map, no torch required.
        L_theta_eta = 0.0
        eps_eff = 0.0
        if not is_pure_legacy and dro_on:
            from ddr_mksvm.kernels.base_kernels import build_kernel
            spec = base_kernel_specs[0]
            kernel_obj = build_kernel(spec)
            kernel_obj.fit(dati)
            L_theta_eta = kernel_obj.lipschitz_bound()
            eps_eff = epsilon

        for l in range(1, L + 1):
            y_hat = np.where(y_label == l, 1.0, -1.0)
            nu_grid = np.logspace(-3, 0, 5) if is_pure_legacy else np.logspace(-6, 0, 7)
            best = train_with_nu_search(K_legacy, y_hat, nu_grid,
                                         epsilon=eps_eff, L_theta_eta=L_theta_eta,
                                         formulation=("legacy_q1" if is_pure_legacy else "ddr_q2"))
            if best is None:
                raise RuntimeError(f"convex subproblem failed for class {l}")
            u_vect[:, l - 1] = best["u"]
            b_vect[l - 1] = best["b"]
            Du_vect[:, l - 1] = y_hat * best["u"]

    else:
        from ddr_mksvm.optim.alternating_trainer import AlternatingTrainer
        net_seed = seed if seed is not None else np.random.SeedSequence().entropy % (2**31)
        trainer = AlternatingTrainer(base_kernel_specs, in_dim=n, dnn_on=dnn_on,
                                      mkl_on=mkl_on, dro_on=dro_on, epsilon=epsilon,
                                      n_outer=n_outer, n_inner=n_inner, seed=net_seed)
        trainer.fit_one_vs_all(dati, y_label, L)
        for l in range(1, L + 1):
            sol = trainer.final_solutions[l - 1]
            y_hat = np.where(y_label == l, 1.0, -1.0)
            u_vect[:, l - 1] = sol["u"]
            b_vect[l - 1] = sol["b"]
            Du_vect[:, l - 1] = y_hat * sol["u"]

    # ------------------------------------------------------------------
    # TESTING PHASE
    # ------------------------------------------------------------------
    if trainer is None:
        if base_kernel_specs[0]["kind"] == "rbf":
            K_test_all = _compute_kernel_test(dati, Xtest, "gaussian_rbf", None, None, alpha=kernel_param)
        else:
            c_type, d = _parse_kernel(kernel_str)
            K_test_all = _compute_kernel_test(dati, Xtest, "polynomial", c_type, d, c_val=kernel_param)
    else:
        import torch
        with torch.no_grad():
            f = trainer.f_theta
            Zdati = f(torch.tensor(dati, dtype=torch.float32).T).T.numpy() if dnn_on else dati
            ZXtest = f(torch.tensor(Xtest, dtype=torch.float32).T).T.numpy() if dnn_on else Xtest

        # Use the same double-precision kernel accumulation as training.
        Zdati = np.asarray(Zdati, dtype=np.float64)
        ZXtest = np.asarray(ZXtest, dtype=np.float64)

        eta = trainer._eta_numpy()
        K_test_all = np.zeros((Zdati.shape[1], ZXtest.shape[1]))
        for w, spec in zip(eta, base_kernel_specs):
            if w == 0:
                continue
            if spec["kind"] == "poly":
                K_test_all += w * (Zdati.T @ ZXtest + spec["c"]) ** spec["degree"]
            elif spec["kind"] == "linear":
                K_test_all += w * (Zdati.T @ ZXtest)
            elif spec["kind"] == "rbf":
                alpha = spec["alpha"]
                K_test_all += w * np.exp(-((Zdati[:, :, None] - ZXtest[:, None, :]) ** 2).sum(0) / (2 * alpha ** 2))

    fun_vals = K_test_all.T @ Du_vect - b_vect[None, :]
    predicted_class = np.argmax(fun_vals, axis=1) + 1

    miscl_testing = int(np.sum(predicted_class != y_test_all))
    testing_error = miscl_testing / m_test_tot

    return testing_error
