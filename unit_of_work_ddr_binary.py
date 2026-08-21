"""
unit_of_work_ddr_binary.py  (v2)

Aligned with the corrected, fully-parameterized unit_of_work_deterministic_binary.py:
imports its DATASET_CONFIG, _apply_transform, _parse_kernel, _compute_kernel,
_compute_kernel_test DIRECTLY (not reimplemented), so DDR-MKSVM automatically
tracks any future change to the deterministic baseline's dataset registry,
transforms, or kernel parsing -- there is exactly one place those live.

Ablation flags (DDR-MKSVM_spec.md Section 6.6):
  dnn_on : bool  -- learned deep feature extractor f_theta (Definition 3).
                    False -> f_theta = Identity.
  mkl_on : bool  -- mix the dataset's configured kernel with
                    `extra_kernel_specs` via a learned eta (Definition 4).
                    Only takes effect if extra_kernel_specs is non-empty.
  dro_on : bool  -- Theorem 1's Wasserstein-DRO penalty
                    eps * L_theta_eta * ||w||_H.

dnn_on=False, mkl_on=False, dro_on=False reproduces
unit_of_work_deterministic_binary.py's numbers EXACTLY for any dataset in
DATASET_CONFIG (Proposition 4.8) -- this is the fast/legacy path below,
which builds the kernel via the SAME kernel_spec_from_config() call as
every other path, so there's no separate hand-copied kernel formula to
drift out of sync.
"""

import numpy as np

from holdouts_train_test import holdouts_train_test
from unit_of_work_deterministic_binary import (
    DATASET_CONFIG,
    _apply_transform,
    _parse_kernel,
    _compute_kernel,
    _compute_kernel_test,
)
from ddr_mksvm.optim.convex_subproblem import train_with_nu_search
from ddr_mksvm.config_adapter import kernel_spec_from_config


def unit_of_work_ddr_binary(DATA, dataset_name="mammographicmass_binary",
                             dnn_on=True, mkl_on=True, dro_on=True,
                             epsilon=0.001, extra_kernel_specs=None,
                             n_outer=6, n_inner=15, seed=None):
    """
    extra_kernel_specs : optional list[dict], additional ddr_mksvm base
        kernels (e.g. [dict(kind='rbf', alpha=1.0)]) to mix in alongside
        the dataset's own configured kernel when mkl_on=True. Ignored
        (and mkl_on forced False) if empty/None -- MKL needs >=2 kernels.
    seed : int or None
        If given, makes THIS call fully reproducible: the same int drives
        BOTH the train/test split (random_state passed to
        holdouts_train_test) AND the deep feature extractor's weight
        initialization (torch.manual_seed inside AlternatingTrainer).
        Default None preserves the original uncontrolled-randomness
        behavior (fresh split + fixed network-init seed=0 every call --
        see main_ddr_binary.py's --seeded flag for why you'd want this
        set to the run index instead).
    """
    if dataset_name not in DATASET_CONFIG:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(DATASET_CONFIG.keys())}")

    config = DATASET_CONFIG[dataset_name]
    transform_type = config["data_transform"]
    kernel_str = config["kernel"]

    # ------------------------------------------------------------------
    # Transform + holdout split (identical to the deterministic script)
    # ------------------------------------------------------------------
    DATA_transformed = _apply_transform(DATA, transform_type)

    testingsamplesize = 0.25
    Atrain, Atest, Btrain, Btest = holdouts_train_test(DATA_transformed, testingsamplesize,
                                                        random_state=seed)

    dati_set_A = Atrain.T
    dati_set_B = Btrain.T
    m_A = dati_set_A.shape[1]
    m_B = dati_set_B.shape[1]
    dati = np.hstack([dati_set_A, dati_set_B])
    n, m = dati.shape
    y = np.concatenate([np.ones(m_A), -np.ones(m_B)])

    # ------------------------------------------------------------------
    # Kernel setup (via the deterministic script's own functions)
    # ------------------------------------------------------------------
    base_kernel_specs, K_legacy, kernel_param = kernel_spec_from_config(
        kernel_str, dati, _parse_kernel, _compute_kernel)

    if mkl_on and extra_kernel_specs:
        base_kernel_specs = base_kernel_specs + list(extra_kernel_specs)
    else:
        mkl_on = False  # nothing to mix with fewer than 2 kernels

    is_pure_legacy = not dnn_on and not mkl_on and not dro_on
    trainer = None

    # ------------------------------------------------------------------
    # TRAINING PHASE
    # ------------------------------------------------------------------
    if is_pure_legacy:
        # Bit-for-bit the deterministic baseline's LP (Proposition 4.8).
        best = train_with_nu_search(K_legacy, y, np.logspace(-3, 0, 5),
                                     epsilon=0.0, L_theta_eta=0.0,
                                     formulation="legacy_q1")
        u_opt, b_opt_opt = best["u"], best["b"]

    elif not dnn_on and not mkl_on:
        # DRO-only ablation: single fixed dataset kernel, Theorem-1 penalty
        # active, no torch needed (L_theta_eta is just this kernel's own
        # analytic Lipschitz bound).
        from ddr_mksvm.kernels.base_kernels import build_kernel
        spec = base_kernel_specs[0]
        kernel_obj = build_kernel(spec)
        kernel_obj.fit(dati)
        L_theta_eta = kernel_obj.lipschitz_bound() if dro_on else 0.0
        best = train_with_nu_search(K_legacy, y, np.logspace(-6, 0, 7),
                                     epsilon=(epsilon if dro_on else 0.0),
                                     L_theta_eta=L_theta_eta)
        u_opt, b_opt_opt = best["u"], best["b"]

    else:
        from ddr_mksvm.optim.alternating_trainer import AlternatingTrainer
        # BUGFIX: AlternatingTrainer defaults to a hardcoded seed=0, which
        # (before this fix) meant every one of the 96 "independent" runs
        # got an IDENTICAL network initialization for DNN-based ablations
        # -- only the data split varied. When the caller doesn't ask for
        # reproducibility (seed=None), draw a fresh random net-init seed
        # per call instead of silently reusing 0.
        net_seed = seed if seed is not None else np.random.SeedSequence().entropy % (2**31)
        trainer = AlternatingTrainer(base_kernel_specs, in_dim=n, dnn_on=dnn_on,
                                      mkl_on=mkl_on, dro_on=dro_on, epsilon=epsilon,
                                      n_outer=n_outer, n_inner=n_inner, seed=net_seed)
        trainer.fit(dati, y)
        u_opt, b_opt_opt = trainer.final_solution["u"], trainer.final_solution["b"]

    # ------------------------------------------------------------------
    # TESTING PHASE
    # ------------------------------------------------------------------
    Atest = Atest.T
    Btest = Btest.T
    m_Atest = Atest.shape[1]
    m_Btest = Btest.shape[1]
    Du_opt = y * u_opt

    if trainer is None:
        if base_kernel_specs[0]["kind"] == "rbf":
            K_test_A = _compute_kernel_test(dati, Atest, "gaussian_rbf", None, None, alpha=kernel_param)
            K_test_B = _compute_kernel_test(dati, Btest, "gaussian_rbf", None, None, alpha=kernel_param)
        else:
            c_type, d = _parse_kernel(kernel_str)
            K_test_A = _compute_kernel_test(dati, Atest, "polynomial", c_type, d, c_val=kernel_param)
            K_test_B = _compute_kernel_test(dati, Btest, "polynomial", c_type, d, c_val=kernel_param)
    else:
        import torch
        with torch.no_grad():
            f = trainer.f_theta
            Zdati = f(torch.tensor(dati, dtype=torch.float32).T).T.numpy() if dnn_on else dati
            ZAtest = f(torch.tensor(Atest, dtype=torch.float32).T).T.numpy() if dnn_on else Atest
            ZBtest = f(torch.tensor(Btest, dtype=torch.float32).T).T.numpy() if dnn_on else Btest

        eta = trainer._eta_numpy()
        K_test_A = np.zeros((Zdati.shape[1], ZAtest.shape[1]))
        K_test_B = np.zeros((Zdati.shape[1], ZBtest.shape[1]))
        for w, spec in zip(eta, base_kernel_specs):
            if w == 0:
                continue
            if spec["kind"] == "poly":
                K_test_A += w * (Zdati.T @ ZAtest + spec["c"]) ** spec["degree"]
                K_test_B += w * (Zdati.T @ ZBtest + spec["c"]) ** spec["degree"]
            elif spec["kind"] == "linear":
                K_test_A += w * (Zdati.T @ ZAtest)
                K_test_B += w * (Zdati.T @ ZBtest)
            elif spec["kind"] == "rbf":
                alpha = spec["alpha"]
                K_test_A += w * np.exp(-((Zdati[:, :, None] - ZAtest[:, None, :]) ** 2).sum(0) / (2 * alpha ** 2))
                K_test_B += w * np.exp(-((Zdati[:, :, None] - ZBtest[:, None, :]) ** 2).sum(0) / (2 * alpha ** 2))

    scores_A = K_test_A.T @ Du_opt - b_opt_opt
    scores_B = K_test_B.T @ Du_opt - b_opt_opt

    falsenegative = int(np.sum(scores_A <= 0))
    falsepositive = int(np.sum(scores_B > 0))
    tot_num_misclass_testing = falsenegative + falsepositive
    testing_error = tot_num_misclass_testing / (m_Atest + m_Btest)

    testing_error_A = falsenegative / m_Atest if m_Atest > 0 else 0.0
    testing_error_B = falsepositive / m_Btest if m_Btest > 0 else 0.0

    return testing_error, testing_error_A, testing_error_B
