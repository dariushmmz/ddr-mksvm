"""Source-faithful q=1 robust SVM primitives from the supplied MATLAB code.

The input uncertainty norm ``p`` is distinct from the fixed coefficient
regularizer ``q=1``.  Source defects are preserved only in explicitly named
MATLAB-parity paths and are reported in diagnostics.
"""
from __future__ import annotations

import hashlib
import itertools
import math
import time
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.optimize import linprog
from scipy.special import comb

NU_GRID = tuple(float(v) for v in np.logspace(-3, 0, 5))
RHO_GRID_BINARY = tuple(float(v) for v in np.logspace(-7, -1, 60))
RHO_GRID_MULTICLASS = tuple(float(v) for v in np.logspace(-7, -1, 7))
MODEL_NAMES = ("deterministic", "robust_p1", "robust_p2", "robust_pinf")


@dataclass(frozen=True)
class KernelSpec:
    kind: str
    degree: int = 1
    offset: float = 0.0
    alpha: float | None = None


@dataclass
class BinaryParityModel:
    X: np.ndarray
    y: np.ndarray
    u: np.ndarray
    gamma: float
    xi: np.ndarray
    b: float
    delta: np.ndarray
    kernel: KernelSpec
    positive_class: int | None = None
    diagnostics: dict | None = None

    def decision(self, Xq: np.ndarray) -> np.ndarray:
        return gram(self.X, Xq, self.kernel).T @ (self.y * self.u) - self.b


def array_hash(values: np.ndarray, dtype="<f8") -> str:
    return hashlib.sha256(np.ascontiguousarray(values, dtype=dtype).tobytes()).hexdigest()


def p_constant(n_features: int, p: float) -> float:
    if n_features <= 0 or p < 1:
        raise ValueError("p>=1 and a positive feature dimension are required")
    if math.isinf(p):
        return math.sqrt(n_features)
    if p <= 2:
        return 1.0
    return float(n_features ** ((p - 2.0) / (2.0 * p)))


def gram(X: np.ndarray, Z: np.ndarray | None, spec: KernelSpec) -> np.ndarray:
    X = np.asarray(X, float)
    Z = X if Z is None else np.asarray(Z, float)
    if spec.kind == "poly":
        return (X @ Z.T + float(spec.offset)) ** int(spec.degree)
    if spec.kind == "rbf":
        if spec.alpha is None or spec.alpha <= 0:
            raise ValueError("RBF alpha must be positive")
        dist2 = np.maximum(np.sum(X * X, 1)[:, None] + np.sum(Z * Z, 1)[None, :] - 2 * X @ Z.T, 0.0)
        return np.exp(-dist2 / (2.0 * spec.alpha**2))
    raise ValueError(f"unsupported kernel {spec.kind!r}")


def class_eta(X: np.ndarray, labels: np.ndarray, rho: float) -> dict[int, float]:
    X, labels = np.asarray(X, float), np.asarray(labels)
    if rho < 0:
        raise ValueError("rho must be nonnegative")
    output = {}
    for label in np.unique(labels):
        rows = X[labels == label]
        if len(rows) < 2:
            raise ValueError("MATLAB sample standard deviation needs two class rows")
        output[int(label)] = float(rho * np.max(np.std(rows, axis=0, ddof=1)))
    return output


def polynomial_delta_point(x: np.ndarray, eta: float, p: float, degree: int,
                           offset: float = 0.0) -> float:
    """Paper Proposition-1 radius for one point (correct formula)."""
    x = np.asarray(x, float)
    ce = p_constant(x.size, p) * eta
    if degree == 1:
        return float(ce)
    base = sum(comb(degree, k, exact=True) * np.linalg.norm(x) ** (degree-k) * ce**k
               for k in range(1, degree+1))
    extra = 0.0
    for k in range(1, degree):
        inner = sum(comb(degree-k, j, exact=True) * np.linalg.norm(x) ** (degree-k-j) * ce**j
                    for j in range(1, degree-k+1))
        extra += comb(degree, k, exact=True) * offset**k * inner**2
    return float(math.sqrt(base**2 + extra))


def matlab_polynomial_delta(X: np.ndarray, labels: np.ndarray, rho: float, p: float,
                            degree: int, offset: float) -> np.ndarray:
    """Literal loop semantics, including cumulative c>0 auxiliary state."""
    X, labels = np.asarray(X, float), np.asarray(labels)
    etas = class_eta(X, labels, rho)
    out = np.zeros(len(X))
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        ce = p_constant(X.shape[1], p) * etas[int(label)]
        if degree == 1:
            out[indices] = ce
            continue
        aux_j = 0.0
        aux_k = 0.0
        for index in indices:
            norm = np.linalg.norm(X[index])
            base = sum(comb(degree, k, exact=True) * norm ** (degree-k) * ce**k
                       for k in range(1, degree+1))
            for k in range(1, degree):
                for j in range(1, degree-k+1):
                    aux_j += comb(degree-k, j, exact=True) * norm ** (degree-k-j) * ce**j
                aux_k += comb(degree, k, exact=True) * offset**k * aux_j**2
            out[index] = math.sqrt(base**2 + aux_k**2)
    return out


def uncertainty_delta(X: np.ndarray, labels: np.ndarray, rho: float, p: float,
                      kernel: KernelSpec, matlab_semantics: bool = True) -> np.ndarray:
    X, labels = np.asarray(X, float), np.asarray(labels)
    if rho == 0:
        return np.zeros(len(X))
    etas = class_eta(X, labels, rho)
    if kernel.kind == "rbf":
        C = p_constant(X.shape[1], p)
        return np.asarray([math.sqrt(max(0.0, 2.0 - 2.0 * math.exp(
            -(C * etas[int(label)])**2 / (2.0 * kernel.alpha**2)))) for label in labels])
    if kernel.kind != "poly":
        raise ValueError(kernel.kind)
    if matlab_semantics:
        return matlab_polynomial_delta(X, labels, rho, p, kernel.degree, kernel.offset)
    return np.asarray([polynomial_delta_point(x, etas[int(label)], p, kernel.degree, kernel.offset)
                       for x, label in zip(X, labels)])


def build_q1_lp(K: np.ndarray, y: np.ndarray, nu: float, delta: np.ndarray):
    K, y, delta = np.asarray(K, float), np.asarray(y, float), np.asarray(delta, float)
    m = len(y)
    if K.shape != (m, m) or delta.shape != (m,):
        raise ValueError("incompatible LP shapes")
    # z=[u(m), gamma, xi(m), s(m)] and all inequalities use A_ub z <= b_ub.
    nvar = 3*m + 1
    c = np.zeros(nvar)
    c[m+1:2*m+1] = float(nu)
    c[2*m+1:] = 1.0
    M = np.outer(y, y) * K
    margin = np.zeros((m, nvar))
    margin[:, :m] = -M
    margin[:, m] = y
    margin[:, m+1:2*m+1] = -np.eye(m)
    margin[:, 2*m+1:] = np.outer(delta, np.sqrt(np.clip(np.diag(K), 0.0, None)))
    epigraph = np.zeros((2*m, nvar))
    epigraph[:m, :m] = np.eye(m)
    epigraph[:m, 2*m+1:] = -np.eye(m)
    epigraph[m:, :m] = -np.eye(m)
    epigraph[m:, 2*m+1:] = -np.eye(m)
    A = np.vstack((margin, epigraph))
    b = np.concatenate((-np.ones(m), np.zeros(2*m)))
    bounds = [(None, None)]*(m+1) + [(0.0, None)]*(2*m)
    return c, A, b, bounds


def solve_q1_lp(K: np.ndarray, y: np.ndarray, nu: float, delta: np.ndarray,
                solver: str = "highs") -> tuple[dict, dict]:
    c, A, b, bounds = build_q1_lp(K, y, nu, delta)
    started = time.perf_counter()
    result = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method=solver)
    elapsed = time.perf_counter() - started
    if not result.success:
        raise RuntimeError(f"HiGHS LP failed status={result.status}: {result.message}")
    m = len(y); z = np.asarray(result.x, float)
    u, gamma, xi, s = z[:m], float(z[m]), z[m+1:2*m+1], z[2*m+1:]
    residual = np.asarray(b - A @ z)
    values = dict(u=u, gamma=gamma, xi=xi, s=s)
    diag = dict(solver=f"scipy-linprog-{solver}", solver_status=str(result.message),
                primal_status="optimal", dual_status="available" if getattr(result, "ineqlin", None) is not None else "unknown",
                solver_iterations=int(result.nit), objective_value=float(result.fun), runtime_s=elapsed,
                min_constraint_residual=float(np.min(residual)), max_constraint_violation=float(max(0.0, -np.min(residual))),
                u_l1=float(np.sum(np.abs(u))), sum_xi=float(np.sum(xi)), robust_penalty_scalar=float(
                    np.sqrt(np.clip(np.diag(K), 0.0, None)) @ np.abs(u)), max_delta=float(np.max(delta)))
    return values, diag


def _strip(gamma: float, xi: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float]:
    signed = y * xi
    omega_pos = float(np.max(signed))
    omega_neg = float(np.max(-signed))
    return gamma + 1.0 - omega_neg, gamma - 1.0 + omega_pos, omega_pos, omega_neg


def matlab_binary_threshold(K: np.ndarray, y: np.ndarray, values: dict,
                            delta: np.ndarray, num_points: int = 10_000):
    u, gamma, xi = values["u"], values["gamma"], values["xi"]
    raw = K @ (y*u)
    robust_scalar = float(np.sqrt(np.clip(np.diag(K), 0, None)) @ np.abs(u))
    left, right, omega_pos, omega_neg = _strip(gamma, xi, y)
    best_count, best_b = len(y), gamma
    for b in np.linspace(left, right, num_points):
        count = int(np.sum(y*(-raw+b) + delta*robust_scalar > 0))
        if count < best_count:
            best_count, best_b = count, float(b)
    return best_b, best_count/len(y), dict(threshold_left=left, threshold_right=right,
        threshold_descending=bool(left > right), threshold_points=num_points,
        omega_positive=omega_pos, omega_negative=omega_neg, threshold_source_bug="none")


def matlab_multiclass_threshold(K: np.ndarray, y: np.ndarray, values: dict,
                                delta: np.ndarray, num_points: int = 10_000):
    """Literal robust multiclass threshold loop, including its mixed criteria."""
    u, gamma, xi = values["u"], values["gamma"], values["xi"]
    raw = K @ (y*u)
    robust_scalar = float(np.sqrt(np.clip(np.diag(K), 0, None)) @ np.abs(u))
    left, right, omega_pos, omega_neg = _strip(gamma, xi, y)
    best_count, best_b = len(y), None
    for b in np.linspace(left, right, num_points):
        nominal_count = int(np.sum(y*(-raw+b) > 0))
        if nominal_count < best_count:
            best_count = int(np.sum(y*(-raw+b + delta*robust_scalar) > 0))
            best_b = float(b)
    if best_b is None:
        raise RuntimeError("MATLAB b_opt_l would be undefined: no strict threshold update")
    training_count = int(np.sum(y*(-raw+best_b + delta*robust_scalar) > 0))
    return best_b, training_count/len(y), dict(threshold_left=left, threshold_right=right,
        threshold_descending=bool(left > right), threshold_points=num_points,
        omega_positive=omega_pos, omega_negative=omega_neg,
        threshold_source_bug="nominal-if/robust-store/delta-inside-D")


def fit_binary(X: np.ndarray, y: np.ndarray, original_labels: np.ndarray,
               kernel: KernelSpec, rho: float, p: float,
               nu_grid: Iterable[float] = NU_GRID, multiclass_bug: bool = False,
               solver: str = "highs") -> tuple[BinaryParityModel, list[dict]]:
    X, y, original_labels = np.asarray(X, float), np.asarray(y, float), np.asarray(original_labels)
    if set(np.unique(y)) != {-1.0, 1.0}:
        raise ValueError("binary y must contain -1 and +1")
    K = gram(X, None, kernel)
    delta = uncertainty_delta(X, original_labels, rho, p, kernel, matlab_semantics=True)
    candidates, diagnostics = [], []
    threshold = matlab_multiclass_threshold if multiclass_bug else matlab_binary_threshold
    for order, nu in enumerate(nu_grid):
        values, diag = solve_q1_lp(K, y, float(nu), delta, solver)
        b, train_error, threshold_diag = threshold(K, y, values, delta)
        diag.update(nu=float(nu), nu_order=order, gamma=values["gamma"], b=b,
                    training_error=float(train_error), rho=float(rho), p="inf" if math.isinf(p) else float(p),
                    delta_hash=array_hash(delta), selected=False, **threshold_diag)
        diagnostics.append(diag)
        candidates.append((train_error, order, values, b, diag))
    chosen = min(candidates, key=lambda row: (row[0], row[1]))
    chosen[4]["selected"] = True
    values, b = chosen[2], chosen[3]
    model = BinaryParityModel(X, y, values["u"], values["gamma"], values["xi"], b,
                              delta, kernel, diagnostics=chosen[4])
    return model, diagnostics


def fit_ova(X: np.ndarray, labels: np.ndarray, kernel: KernelSpec, rho: float, p: float,
            solver: str = "highs") -> tuple[list[BinaryParityModel], list[dict]]:
    models, diagnostics = [], []
    for task, label in enumerate(sorted(np.unique(labels).tolist())):
        y = np.where(labels == label, 1.0, -1.0)
        model, rows = fit_binary(X, y, labels, kernel, rho, p, multiclass_bug=True, solver=solver)
        model.positive_class = int(label)
        for row in rows:
            row.update(task=task, positive_class=int(label))
        models.append(model); diagnostics.extend(rows)
    return models, diagnostics


def predict_ova(models: list[BinaryParityModel], X: np.ndarray,
                tie_policy: str = "matlab_error") -> tuple[np.ndarray, np.ndarray, int]:
    scores = np.column_stack([model.decision(X) for model in models])
    winners = scores == np.max(scores, axis=1, keepdims=True)
    ties = int(np.sum(np.sum(winners, axis=1) > 1))
    if ties and tie_policy == "matlab_error":
        raise RuntimeError(f"MATLAB find/if tie behavior is undefined for {ties} observations")
    if tie_policy not in {"matlab_error", "first"}:
        raise ValueError(tie_policy)
    classes = np.asarray([model.positive_class for model in models])
    return classes[np.argmax(scores, axis=1)], scores, ties
