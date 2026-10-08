"""Conic RKHS-L2 robust counterpart of frozen class-sensitive V8-C."""
from __future__ import annotations

import itertools
import hashlib
import math
import time

import numpy as np
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

from . import v7_cross_dataset as frozen_v7
from .iris_research import ALPHA_RULES, SVC_C_GRID, alpha_from_rule
from .robust_matlab_parity import KernelSpec, gram, uncertainty_delta
from .robust_solvers.feasibility import assess_feasibility
from .robust_solvers.registry import production_policy
from .v8_class_sensitive import fit_selected as frozen_fit_selected
from .v8_class_sensitive import pair_weights, select_family as frozen_select_family, vote


def _factor_psd(matrix: np.ndarray, tolerance: float = 1e-8) -> tuple[np.ndarray, float]:
    matrix = (np.asarray(matrix, float) + np.asarray(matrix, float).T) / 2
    values, vectors = np.linalg.eigh(matrix)
    minimum = float(values.min())
    scale = max(1.0, float(np.max(np.abs(values))))
    if minimum < -tolerance * scale:
        raise ValueError(f"Gram-derived M is materially indefinite: {minimum}")
    values = np.clip(values, 0.0, None)
    return np.sqrt(values)[:, None] * vectors.T, minimum


def solve_pair(X: np.ndarray, y: np.ndarray, C: float, alpha: float, rho: float,
               p: float, solver: str = "CLARABEL",
               representation_mode: str = "deduplicate_exact") -> tuple[dict, dict]:
    try:
        import cvxpy as cp
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Robust V8-C requires CVXPY") from exc
    X, y = np.asarray(X, float), np.asarray(y)
    classes = np.unique(y)
    if len(classes) != 2:
        raise ValueError("one pair must contain exactly two classes")
    signed = np.where(y == classes[0], -1.0, 1.0)
    kernel = KernelSpec("rbf", alpha=float(alpha))
    sample_K = gram(X, None, kernel)
    sample_delta = uncertainty_delta(X, y, float(rho), float(p), kernel, matlab_semantics=False)
    weights_by_class = pair_weights(y, "sqrt")
    if representation_mode == "sample":
        basis_X = X
        constraint_y = y
        constraint_signed = signed
        multiplicities = np.ones(len(y), dtype=int)
        coefficient_multiplier = signed
        score_matrix = sample_K
        delta = sample_delta
        weights = np.asarray([weights_by_class[value] for value in y], float)
        M = np.outer(signed, signed) * sample_K
    elif representation_mode == "deduplicate_exact":
        basis_X, location = np.unique(X, axis=0, return_inverse=True)
        group_keys = sorted(set(zip(location.tolist(), y.tolist())))
        group_location = np.asarray([key[0] for key in group_keys], int)
        constraint_y = np.asarray([key[1] for key in group_keys], dtype=y.dtype)
        constraint_signed = np.where(constraint_y == classes[0], -1.0, 1.0)
        multiplicities = np.asarray([
            np.sum((location == loc) & (y == label)) for loc, label in group_keys
        ], int)
        representatives = np.asarray([
            np.flatnonzero((location == loc) & (y == label))[0] for loc, label in group_keys
        ], int)
        coefficient_multiplier = np.ones(len(basis_X), float)
        score_matrix = gram(basis_X, basis_X[group_location], kernel).T
        delta = sample_delta[representatives]
        weights = multiplicities * np.asarray([weights_by_class[value] for value in constraint_y], float)
        M = gram(basis_X, None, kernel)
    else:
        raise ValueError(f"unknown representation_mode: {representation_mode}")
    H, min_eigenvalue = _factor_psd(M)
    a = cp.Variable(len(basis_X)); intercept = cp.Variable(); xi = cp.Variable(len(constraint_y), nonneg=True); t = cp.Variable(nonneg=True)
    scores = score_matrix @ cp.multiply(coefficient_multiplier, a) + intercept
    constraints = [cp.multiply(constraint_signed, scores) - delta*t >= 1-xi, cp.norm(H @ a, 2) <= t]
    problem = cp.Problem(cp.Minimize(0.5*cp.square(t) + float(C)*(weights @ xi)), constraints)
    started = time.perf_counter()
    policy = production_policy()
    attempt = policy.attempts[0]
    kwargs = dict(attempt.options) if solver == attempt.solver else {}
    problem.solve(solver=solver, **kwargs)
    elapsed = time.perf_counter()-started
    if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or a.value is None:
        raise RuntimeError(f"robust V8-C pair solve failed: {problem.status}")
    coeff = np.asarray(a.value, float); b = float(intercept.value); slack = np.asarray(xi.value, float); norm_bound = float(t.value)
    raw = score_matrix @ (coefficient_multiplier*coeff) + b
    margin_residual = constraint_signed*raw-delta*norm_bound-1+slack
    cone_residual = norm_bound-np.linalg.norm(H@coeff)
    iterations = problem.solver_stats.num_iters
    model = {"X": basis_X, "classes": classes, "signed": coefficient_multiplier,
             "coefficient_multiplier": coefficient_multiplier, "coeff": coeff, "intercept": b,
             "alpha": float(alpha), "delta": delta, "norm_bound": norm_bound,
             "representation_mode": representation_mode}
    diag = {"pair": [int(classes[0]), int(classes[1])], "counts": {str(c): int(np.sum(y==c)) for c in classes},
        "weights": {str(k): float(v) for k,v in weights_by_class.items()}, "C": float(C), "alpha": float(alpha),
        "p": "inf" if math.isinf(p) else float(p), "rho": float(rho), "solver": solver,
        "solver_status": problem.status, "primal_status": problem.status, "dual_status": "available",
        "solver_iterations": int(iterations) if iterations is not None else -1, "runtime_s": elapsed,
        "objective_value": float(problem.value), "rkhs_norm_bound": norm_bound,
        "regularization_term": 0.5*norm_bound**2, "weighted_slack_term": float(C)*(weights@slack),
        "max_delta": float(np.max(delta)), "mean_delta": float(np.mean(delta)),
        "min_margin_residual": float(np.min(margin_residual)), "cone_residual": float(cone_residual),
        "max_constraint_violation": float(max(0.0, -np.min(margin_residual), -cone_residual)),
        "min_M_eigenvalue_before_clip": min_eigenvalue, "intercept": b,
        "representation_mode": representation_mode, "sample_count": int(len(y)),
        "unique_feature_locations": int(len(basis_X)),
        "constraint_group_count": int(len(constraint_y)),
        "maximum_group_multiplicity": int(np.max(multiplicities)),
        "coefficient_vector": coeff.tolist(),
        "coefficient_hash": hashlib.sha256(np.asarray(coeff,dtype="<f8").tobytes()).hexdigest(),
        "training_X_hash": hashlib.sha256(np.asarray(X,dtype="<f8").tobytes()).hexdigest()}
    assessment = assess_feasibility(problem.status, diag, policy.thresholds)
    diag["feasibility_assessment"] = assessment
    diag["solver_attempt"] = attempt.name if solver == attempt.solver else f"explicit:{solver}"
    if not assessment["accepted"]:
        raise RuntimeError(f"robust V8-C pair rejected by feasibility policy: {assessment['reasons']}")
    return model, diag


def predict_pair(model: dict, X: np.ndarray) -> np.ndarray:
    kernel = KernelSpec("rbf", alpha=model["alpha"])
    multiplier = model.get("coefficient_multiplier", model["signed"])
    return gram(model["X"], X, kernel).T @ (multiplier*model["coeff"]) + model["intercept"]


def fit_model(X: np.ndarray, y: np.ndarray, C: float, alpha: float, rho: float, p: float):
    classes = np.unique(y); models = []; diagnostics = []
    for first, second in itertools.combinations(classes, 2):
        mask = (y == first) | (y == second)
        model, diag = solve_pair(X[mask], y[mask], C, alpha, rho, p)
        models.append(model); diagnostics.append(diag)
    return {"classes": classes, "models": models, "diagnostics": diagnostics}


def predict_model(model: dict, X: np.ndarray):
    margins = np.column_stack([predict_pair(pair, X) for pair in model["models"]])
    return (*vote(model["classes"], margins), margins)


def select_and_fit(X: np.ndarray, y: np.ndarray, Z: np.ndarray, transform: str,
                   seed: int, rho: float, p: float):
    """Training-only robust alpha/C selection; rho=0 dispatches frozen V8-C."""
    if rho == 0:
        records, folds, elapsed = frozen_select_family(X, y, transform, seed, "sqrt")
        pred, params, trace, runtime = frozen_fit_selected(X, y, Z, transform, "sqrt", "balanced", records, elapsed)
        params.update(rho=0.0, p="inf" if math.isinf(p) else float(p), reduction="exact frozen V8-C dispatch")
        return pred, params, trace, runtime
    started = time.perf_counter()
    folds = list(StratifiedKFold(3, shuffle=True, random_state=20000+int(seed)).split(X, y))
    prepared = []
    for fi, va in folds:
        A, B, prep = frozen_v7.fit_transform(X[fi], X[va], transform)
        prepared.append((A, B, y[fi], y[va], prep))
    bank = []
    for ai, rule in enumerate(ALPHA_RULES):
        for ci, C in enumerate(SVC_C_GRID):
            losses, fold_diags = [], []
            for fold, (A, B, ya, yb, _) in enumerate(prepared):
                alpha = alpha_from_rule(A, rule)
                fitted = fit_model(A, ya, float(C), alpha, rho, p)
                pred, votes, margins = predict_model(fitted, B)
                losses.append(float(1-balanced_accuracy_score(yb, pred)))
                fold_diags.append({"fold": fold, "alpha": alpha, "loss": losses[-1],
                                   "solver_diagnostics": fitted["diagnostics"]})
            bank.append({"alpha_rule": rule, "C": float(C), "order": [ai,ci],
                         "balanced_loss": float(np.mean(losses)), "folds": fold_diags})
    selected = min(bank, key=lambda row: (row["balanced_loss"], *row["order"]))
    A, B, prep = frozen_v7.fit_transform(X, Z, transform)
    alpha = alpha_from_rule(A, selected["alpha_rule"])
    fitted = fit_model(A, y, selected["C"], alpha, rho, p)
    pred, votes, margins = predict_model(fitted, B)
    params = {"alpha_rule": selected["alpha_rule"], "alpha": alpha, "C": selected["C"],
              "balanced_loss": selected["balanced_loss"], "order": selected["order"],
              "rho": float(rho), "p": "inf" if math.isinf(p) else float(p),
              "family": "sqrt", "preprocessing": prep, "reduction": "conic robust RKHS-L2"}
    trace = {"bank": bank, "solver_diagnostics": fitted["diagnostics"],
             "margins_positive_second": margins.tolist(), "votes": votes.tolist(),
             "vote_ties": int(np.sum((votes==votes.max(axis=1)[:,None]).sum(axis=1)>1))}
    return pred, params, trace, time.perf_counter()-started
