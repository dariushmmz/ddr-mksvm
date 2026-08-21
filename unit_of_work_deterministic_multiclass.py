"""
Python translation of unit_of_work_deterministic_multiclass.m

Deterministic multiclass SVM-type classifier using one-vs-all strategy.
FULLY PARAMETERIZED: supports all multiclass datasets from the paper
with their specific data transforms and kernels.

Multiclass datasets (from Paper Table 3 + user corrections):
  - iris:              none,           gaussian_rbf,     3 classes
  - wine:              standardization, inhom_linear,    3 classes
  - heart_disease:     standardization, inhom_linear,    5 classes
  - dermatology:       none,           inhom_quadratic,  6 classes

Results saved to results/<dataset-name>/deterministic/ directory.

Returns
-------
testing_error : overall multiclass misclassification rate on the test set
"""

import math
import os
import numpy as np
import cvxpy as cp
import pandas as pd

from holdouts_train_test_multiclass import holdouts_train_test_multiclass


# ------------------------------------------------------------------------------
# Dataset configuration registry (from Paper Table 3 + user corrections)
# ------------------------------------------------------------------------------
DATASET_CONFIG = {
    "iris": {
        "csv_name": "iris_multiclass.csv",
        "data_transform": "none",
        "kernel": "gaussian_rbf",
        "classes": 3,
    },
    "wine": {
        "csv_name": "wine.csv",
        "data_transform": "standardization",
        "kernel": "inhom_linear",
        "classes": 3,
    },
    "heart_disease": {
        "csv_name": "heart_disease.csv",
        "data_transform": "standardization",
        "kernel": "inhom_linear",
        "classes": 5,
    },
    "dermatology": {
        "csv_name": "dermatology.csv",
        "data_transform": "none",
        "kernel": "inhom_quadratic",
        "classes": 6,
    },
}


def _apply_transform(DATA, transform_type):
    """Apply the specified data transformation to feature columns."""
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


def unit_of_work_deterministic_multiclass(DATA, dataset_name="iris", seed=None):
    """
    Main function: train and test one deterministic multiclass SVM instance.

    Parameters
    ----------
    DATA : pd.DataFrame
        Raw data with features in all columns except last, label in last column.
        Labels must be integers 1..L (will be auto-converted if 0-indexed).
    dataset_name : str
        Must be a key in DATASET_CONFIG. Determines transform and kernel.
    seed : int or None
        See unit_of_work_deterministic_binary.py's docstring -- same
        pairing rationale, passed through to holdouts_train_test_multiclass.
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
    expected_classes = config["classes"]

    # ------------------------------------------------------------------
    # Apply data transformation
    # ------------------------------------------------------------------
    DATA_transformed = _apply_transform(DATA, transform_type)

    # Ensure CLASS column exists for holdouts_train_test_multiclass
    # Rename last column to CLASS if it isn't already
    if DATA_transformed.columns[-1] != "CLASS":
        cols = list(DATA_transformed.columns)
        cols[-1] = "CLASS"
        DATA_transformed.columns = cols

    # ------------------------------------------------------------------
    # Training vs testing sets
    # ------------------------------------------------------------------
    testingsamplesize = 0.25
    DATAtrain, DATAtest = holdouts_train_test_multiclass(DATA_transformed, testingsamplesize,
                                                          random_state=seed)

    # class label is the last column (1..L integer labels)
    y_label = DATAtrain[:, -1].astype(int)
    dati = DATAtrain[:, :-1].T          # shape (n, m_train)
    n, m_train_tot = dati.shape

    Xtest = DATAtest[:, :-1].T          # shape (n, m_test)
    y_test_all = DATAtest[:, -1].astype(int)
    m_test_tot = Xtest.shape[1]

    # Auto-detect number of classes from data (in case it differs from expected)
    L = int(max(y_label.max(), y_test_all.max()))
    if L != expected_classes:
        print(f"  WARNING: Expected {expected_classes} classes, found {L} in data.")

    # ------------------------------------------------------------------
    # Kernel setup
    # ------------------------------------------------------------------
    if kernel_str == "gaussian_rbf":
        kernel_type = "gaussian_rbf"
        c_type = None
        d = None
        c_val = None
        K, alpha = _compute_kernel(dati, kernel_type, c_type, d)
    else:
        kernel_type = "polynomial"
        c_type, d = _parse_kernel(kernel_str)
        K, c_val = _compute_kernel(dati, kernel_type, c_type, d)

    ones_m = np.ones(m_train_tot)

    # ------------------------------------------------------------------
    # TRAINING PHASE (one-vs-all, L binary sub-problems)
    # ------------------------------------------------------------------
    vectornu = np.logspace(-3, 0, 5)

    u_vect = np.zeros((m_train_tot, L))
    b_vect = np.zeros(L)
    Du_vect = np.zeros((m_train_tot, L))

    for l in range(1, L + 1):
        y_hat = np.where(y_label == l, 1.0, -1.0)

        M = np.outer(y_hat, y_hat) * K

        training_error_opt = np.inf
        u_opt_l = None
        b_opt_opt_l = None

        for nu in vectornu:
            u_l = cp.Variable(m_train_tot)
            vargamma_l = cp.Variable()
            xi_l = cp.Variable(m_train_tot)
            s_l = cp.Variable(m_train_tot)

            objective = cp.Minimize(cp.sum(s_l) + nu * cp.sum(xi_l))

            constraints = [
                M @ u_l - y_hat * vargamma_l + xi_l >= ones_m,
                xi_l >= 0,
                u_l >= -s_l,
                u_l <= s_l,
                s_l >= 0,
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
            print(f"  [class={l}, nu={nu:.4f}] solved with {solver_used}, status={problem.status}")

            if u_l.value is None:
                continue

            u_val = u_l.value
            vargamma_val = float(vargamma_l.value)
            xi_val = xi_l.value

            Dxi = y_hat * xi_val
            omega_minus_l = -np.min(Dxi)
            omega_l = np.max(Dxi)

            num_points = 10000
            discr_b_l = np.linspace(vargamma_val + 1 - omega_minus_l,
                                     vargamma_val - 1 + omega_l,
                                     num_points)

            base = -(M @ u_val)
            counts = np.sum((base[None, :] + y_hat[None, :] * discr_b_l[:, None]) > 0, axis=1)
            best_idx = np.argmin(counts)
            max_b = counts[best_idx]
            b_opt_l = discr_b_l[best_idx] if max_b < m_train_tot else vargamma_val

            tot_num_misclass_training = np.sum((base + y_hat * b_opt_l) > 0)
            training_error = tot_num_misclass_training / m_train_tot

            if training_error < training_error_opt:
                training_error_opt = training_error
                u_opt_l = u_val
                b_opt_opt_l = b_opt_l

        u_vect[:, l - 1] = u_opt_l
        b_vect[l - 1] = b_opt_opt_l
        Du_vect[:, l - 1] = y_hat * u_opt_l

    # ------------------------------------------------------------------
    # TESTING PHASE
    # ------------------------------------------------------------------
    if kernel_type == "polynomial":
        K_test_all = _compute_kernel_test(dati, Xtest, kernel_type, c_type, d, c_val)
    elif kernel_type == "gaussian_rbf":
        K_test_all = _compute_kernel_test(dati, Xtest, kernel_type, None, None, alpha=alpha)

    fun_vals = K_test_all.T @ Du_vect - b_vect[None, :]
    predicted_class = np.argmax(fun_vals, axis=1) + 1

    miscl_testing = int(np.sum(predicted_class != y_test_all))
    testing_error = miscl_testing / m_test_tot

    return testing_error