"""
Python translation of unit_of_work_deterministic_binary.m

Deterministic (non-robust) binary SVM-type classifier.
Same kernel/training/testing structure as the robust version,
but WITHOUT perturbation modeling (no rho, no delta, no C-scaling).

FULLY PARAMETERIZED: supports all 10 binary datasets from the paper
with their specific data transforms and kernels.

Dataset configurations (from Table 3):
  - arrhythmia:           none,           gaussian_rbf
  - parkinson:            minmax,         hom_linear
  - heart_disease:        standardization, inhom_linear
  - dermatology:          none,           inhom_quadratic
  - climate_model_crashes: none,          hom_linear
  - breast_cancer_diagnostic: minmax,     inhom_quadratic
  - breast_cancer:        standardization, hom_linear
  - blood_transfusion:    standardization, inhom_cubic
  - mammographicmass_binary: standardization, inhom_quadratic
  - qsar_biodegradation:  minmax,         gaussian_rbf

Results saved to results/<dataset-name>/ directory.

Returns
-------
testing_error   : overall misclassification rate on the test set
testing_error_A : misclassification rate on class-A (false negatives)
testing_error_B : misclassification rate on class-B (false positives)
"""

import math
import os
import numpy as np
import cvxpy as cp
import pandas as pd

from holdouts_train_test import holdouts_train_test


# ------------------------------------------------------------------------------
# Dataset configuration registry (from Paper Table 3)
# ------------------------------------------------------------------------------
DATASET_CONFIG = {
    # 8 binary datasets from the paper (Table 3)
    "arrhythmia": {
        "csv_name": "arrhythmia.csv",
        "data_transform": "none",
        "kernel": "gaussian_rbf",
    },
    "parkinson": {
        "csv_name": "parkinson.csv",
        "data_transform": "minmax",
        "kernel": "hom_linear",
    },
    # NOTE: "heart_disease" and "dermatology" are MULTICLASS datasets,
    # not binary. Use the multiclass scripts for those.
    "climate_model_crashes": {
        "csv_name": "climate_model_crashes.csv",
        "data_transform": "none",
        "kernel": "hom_linear",
    },
    "breast_cancer_diagnostic": {
        "csv_name": "breast_cancer_diagnostic.csv",
        "data_transform": "minmax",
        "kernel": "inhom_quadratic",
    },
    "breast_cancer": {
        "csv_name": "breast_cancer.csv",
        "data_transform": "standardization",
        "kernel": "hom_linear",
    },
    "blood_transfusion": {
        "csv_name": "blood_transfusion.csv",
        "data_transform": "standardization",
        "kernel": "inhom_cubic",
    },
    "mammographicmass_binary": {
        "csv_name": "mammographicmass_binary.csv",
        "data_transform": "standardization",
        "kernel": "inhom_quadratic",
    },
    "qsar_biodegradation": {
        "csv_name": "qsar_biodegradation.csv",
        "data_transform": "minmax",
        "kernel": "gaussian_rbf",
    },
}


def prepare_binary_data(DATA):
    """Coerce binary data to finite numeric values and mean-impute features.

    The raw Mammographic Mass dataset contains missing feature values (often
    encoded as ``?``).  Keeping all 961 rows therefore requires an explicit
    missing-value policy before standardization and kernel construction.

    Returns
    -------
    clean_data : pandas.DataFrame
        Numeric data with finite, mean-imputed feature columns.
    imputed_counts : dict[str, int]
        Number of missing/non-numeric/non-finite values replaced per feature.
    """
    if not isinstance(DATA, pd.DataFrame) or DATA.shape[1] < 2:
        raise ValueError("DATA must be a DataFrame with feature columns and a final label column")

    feature_names = list(DATA.columns[:-1])
    label_name = DATA.columns[-1]
    features = DATA.iloc[:, :-1].apply(pd.to_numeric, errors="coerce")
    features = features.replace([np.inf, -np.inf], np.nan)
    missing_counts = features.isna().sum()

    all_missing = [str(name) for name in feature_names if features[name].isna().all()]
    if all_missing:
        raise ValueError(
            "Feature columns contain no usable numeric values: " + ", ".join(all_missing)
        )

    # Mean imputation is consistent with the existing global standardization:
    # after centering, an imputed value maps exactly to zero.
    features = features.fillna(features.mean(axis=0))
    labels = pd.to_numeric(DATA.iloc[:, -1], errors="coerce")
    invalid_labels = labels.isna() | ~np.isfinite(labels.to_numpy(dtype=float))
    if invalid_labels.any():
        rows = DATA.index[invalid_labels].tolist()[:10]
        raise ValueError(f"Label column '{label_name}' has missing/non-numeric values at rows {rows}")

    unique_labels = set(labels.astype(float).unique())
    if unique_labels != {0.0, 1.0}:
        raise ValueError(
            f"Binary label column '{label_name}' must contain exactly 0 and 1; "
            f"found {sorted(unique_labels)}"
        )

    clean_data = features.copy()
    clean_data[label_name] = labels.to_numpy(dtype=float)
    return clean_data, {
        str(name): int(count) for name, count in missing_counts.items() if count
    }


def _apply_transform(DATA, transform_type):
    """Apply the specified data transformation to feature columns."""
    DATA, _ = prepare_binary_data(DATA)
    data_array = DATA.values
    feature_cols = data_array[:, :-1].astype(float)
    label_col = data_array[:, -1]

    if transform_type == "standardization":
        mean_vec = np.mean(feature_cols, axis=0)
        std_vec = np.std(feature_cols, axis=0, ddof=0)
        std_vec[std_vec == 0] = 1.0
        transformed_features = (feature_cols - mean_vec) / std_vec

    elif transform_type == "minmax":
        min_vec = np.min(feature_cols, axis=0)
        max_vec = np.max(feature_cols, axis=0)
        range_vec = max_vec - min_vec
        range_vec[range_vec == 0] = 1.0
        transformed_features = (feature_cols - min_vec) / range_vec

    elif transform_type == "none":
        transformed_features = feature_cols

    else:
        raise ValueError(f"Unknown transform type: {transform_type}")

    transformed_data = np.column_stack([transformed_features, label_col])
    return pd.DataFrame(transformed_data, columns=DATA.columns)


def _parse_kernel(kernel_str):
    """Parse kernel string into (kernel_type, degree, c_type)."""
    parts = kernel_str.split("_")
    if len(parts) != 2:
        raise ValueError(f"Invalid kernel string: {kernel_str}")

    c_type = parts[0]   # "hom" or "inhom"
    degree_str = parts[1]

    if degree_str == "linear":
        d = 1
    elif degree_str == "quadratic":
        d = 2
    elif degree_str == "cubic":
        d = 3
    elif degree_str == "rbf":
        d = None
    else:
        raise ValueError(f"Unknown degree in kernel string: {kernel_str}")

    return c_type, d


def _compute_kernel(dati, kernel_type, c_type, d, c_val=None, alpha=None):
    """Compute the Gram matrix K for the specified kernel."""
    if kernel_type == "polynomial":
        if c_type == "inhom":
            c = c_val if c_val is not None else float(np.max(np.std(dati, axis=1, ddof=0)))
        else:
            c = 0.0
        K = (dati.T @ dati + c) ** d
        return K, c

    elif kernel_type == "gaussian_rbf":
        if alpha is None:
            alpha = float(np.max(np.std(dati, axis=1, ddof=0)))
        diff = dati[:, :, None] - dati[:, None, :]
        sqdist = np.sum(diff ** 2, axis=0)
        K = np.exp(-sqdist / (2 * alpha ** 2))
        return K, alpha

    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}")


def _compute_kernel_test(dati, Xtest, kernel_type, c_type, d, c_val=None, alpha=None):
    """Compute kernel matrix between training data and test data."""
    if kernel_type == "polynomial":
        if c_type == "inhom":
            c = c_val if c_val is not None else float(np.max(np.std(dati, axis=1, ddof=0)))
        else:
            c = 0.0
        K_test = (dati.T @ Xtest + c) ** d
        return K_test

    elif kernel_type == "gaussian_rbf":
        if alpha is None:
            alpha = float(np.max(np.std(dati, axis=1, ddof=0)))
        diff = dati[:, :, None] - Xtest[:, None, :]
        sqdist = np.sum(diff ** 2, axis=0)
        K_test = np.exp(-sqdist / (2 * alpha ** 2))
        return K_test

    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}")


def unit_of_work_deterministic_binary(DATA, dataset_name="mammographicmass_binary", seed=None):
    """
    Main function: train and test one deterministic SVM instance.

    Parameters
    ----------
    DATA : pd.DataFrame
        Raw data with features in all columns except last, label in last column.
    dataset_name : str
        Must be a key in DATASET_CONFIG. Determines transform and kernel.
    seed : int or None
        Passed as random_state to holdouts_train_test. Default None
        preserves the original uncontrolled-random-split behavior (the
        96-independent-runs protocol). Pass the run index to make this
        call's split reproducible -- needed to PAIR each run against the
        corresponding run of unit_of_work_ddr_binary(..., seed=i) for a
        fair, matched-split comparison instead of two independent samples
        of 96 splits.
    """
    # ------------------------------------------------------------------
    # Lookup dataset configuration
    # ------------------------------------------------------------------
    if dataset_name not in DATASET_CONFIG:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            f"Available: {list(DATASET_CONFIG.keys())}"
        )

    config = DATASET_CONFIG[dataset_name]
    transform_type = config["data_transform"]
    kernel_str = config["kernel"]

    # ------------------------------------------------------------------
    # Apply data transformation
    # ------------------------------------------------------------------
    DATA_transformed = _apply_transform(DATA, transform_type)

    # ------------------------------------------------------------------
    # Training vs testing sets
    # ------------------------------------------------------------------
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
    # Kernel setup
    # ------------------------------------------------------------------
    if kernel_str == "gaussian_rbf":
        kernel_type = "gaussian_rbf"
        c_type = None
        d = None
        c_val = None
        K, alpha = _compute_kernel(dati, kernel_type, c_type, d)
        kernel_param = alpha
    else:
        kernel_type = "polynomial"
        c_type, d = _parse_kernel(kernel_str)
        K, c_val = _compute_kernel(dati, kernel_type, c_type, d)
        kernel_param = c_val

    # ------------------------------------------------------------------
    # TRAINING PHASE (no perturbation / no delta)
    # ------------------------------------------------------------------
    vectornu = np.logspace(-3, 0, 5)

    training_error_opt = np.inf
    u_opt = None
    b_opt_opt = None

    ones_m = np.ones(m)
    M = np.outer(y, y) * K

    for nu in vectornu:
        u = cp.Variable(m)
        vargamma = cp.Variable()
        xi = cp.Variable(m)
        s = cp.Variable(m)

        objective = cp.Minimize(cp.sum(s) + nu * cp.sum(xi))

        # NOTE: no delta term here (deterministic, no robustness)
        constraints = [
            M @ u - y * vargamma + xi >= ones_m,
            xi >= 0,
            u >= -s,
            u <= s,
            s >= 0,
        ]

        problem = cp.Problem(objective, constraints)
        solver_used = None
        try:
            problem.solve(solver=cp.HIGHS)
            solver_used = "HIGHS"
        except Exception:
            try:
                problem.solve(solver=cp.CLARABEL)
                solver_used = "CLARABEL"
            except Exception:
                try:
                    problem.solve(solver=cp.SCS)
                    solver_used = "SCS"
                except Exception:
                    problem.solve()
                    solver_used = "default"
        print(f"  [nu={nu:.4f}] solved with {solver_used}, status={problem.status}")

        if u.value is None:
            continue

        u_val = u.value
        vargamma_val = float(vargamma.value)
        xi_val = xi.value

        Dxi = y * xi_val
        omega_A = np.max(Dxi)
        omega_B = np.max(-Dxi)

        num_points = 10000
        discr_b = np.linspace(vargamma_val + 1 - omega_B,
                               vargamma_val - 1 + omega_A,
                               num_points)

        # No delta term in deterministic version
        base = -(M @ u_val)
        counts = np.sum((base[None, :] + y[None, :] * discr_b[:, None]) > 0, axis=1)
        best_idx = np.argmin(counts)
        max_b = counts[best_idx]
        b_opt = discr_b[best_idx] if max_b < m else vargamma_val

        tot_num_misclass_training = np.sum((-(M @ u_val) + y * b_opt) > 0)
        training_error = tot_num_misclass_training / m

        if training_error < training_error_opt:
            training_error_opt = training_error
            u_opt = u_val
            b_opt_opt = b_opt

    # ------------------------------------------------------------------
    # TESTING PHASE (no perturbation / no delta)
    # ------------------------------------------------------------------
    Atest = Atest.T
    Btest = Btest.T

    m_Atest = Atest.shape[1]
    m_Btest = Btest.shape[1]

    Du_opt = y * u_opt

    if kernel_type == "polynomial":
        K_test_A = _compute_kernel_test(dati, Atest, kernel_type, c_type, d, c_val)
        K_test_B = _compute_kernel_test(dati, Btest, kernel_type, c_type, d, c_val)
    elif kernel_type == "gaussian_rbf":
        K_test_A = _compute_kernel_test(dati, Atest, kernel_type, None, None, alpha=alpha)
        K_test_B = _compute_kernel_test(dati, Btest, kernel_type, None, None, alpha=alpha)

    scores_A = K_test_A.T @ Du_opt - b_opt_opt
    scores_B = K_test_B.T @ Du_opt - b_opt_opt

    falsenegative = int(np.sum(scores_A <= 0))
    falsepositive = int(np.sum(scores_B > 0))

    tot_num_misclass_testing = falsenegative + falsepositive
    testing_error = tot_num_misclass_testing / (m_Atest + m_Btest)

    # Guard against empty test sets (can happen with unbalanced splits)
    testing_error_A = falsenegative / m_Atest if m_Atest > 0 else 0.0
    testing_error_B = falsepositive / m_Btest if m_Btest > 0 else 0.0

    return testing_error, testing_error_A, testing_error_B
