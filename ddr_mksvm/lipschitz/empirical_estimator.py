"""
Empirical Lipschitz constant estimator (Section 6.4 of the spec):
random-pair sampling followed by local gradient ascent on the ratio
||f(x)-f(x')|| / ||x-x'||.

Used ONLY for diagnostics: at every outer training iteration, log both
this empirical estimate and Lemma 2's analytic bound, and warn if their
ratio exceeds ~10x (the analytic bound has become too loose to be a
useful DRO penalty coefficient). Never use this estimate as the L_f fed
into Theorem 1's penalty term itself -- it is not a *certified* bound
(gradient ascent can undershoot the true supremum), only a lower-bound
sanity check on the certified analytic one.
"""

import numpy as np

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TORCH_AVAILABLE = False


def empirical_lipschitz(f, X, n_pairs=2000, refine_steps=20, lr=0.05, seed=None):
    """
    f : callable, torch.Tensor (n_pairs, in_dim) -> torch.Tensor (n_pairs, out_dim)
    X : ndarray (m, in_dim) -- pool of points to sample pairs from
    """
    if not _TORCH_AVAILABLE:
        raise ImportError("torch is required for empirical_lipschitz.")

    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=float)
    m = X.shape[0]
    if m < 2:
        raise ValueError("Need at least 2 points to estimate a Lipschitz constant.")

    idx1 = rng.integers(0, m, n_pairs)
    idx2 = rng.integers(0, m, n_pairs)
    x1 = torch.tensor(X[idx1], dtype=torch.float32)
    x2 = torch.tensor(X[idx2], dtype=torch.float32)

    with torch.no_grad():
        y1, y2 = f(x1), f(x2)
        num = (y1 - y2).norm(dim=1)
        den = (x1 - x2).norm(dim=1).clamp_min(1e-8)
        ratios = num / den

    best_ratio = float(ratios.max().item())
    best_i = int(torch.argmax(ratios).item())

    # local refinement: gradient ascent on the ratio around the best sampled pair
    xa = x1[best_i:best_i + 1].clone().detach().requires_grad_(True)
    xb = x2[best_i:best_i + 1].clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([xa, xb], lr=lr)
    for _ in range(refine_steps):
        opt.zero_grad()
        ya, yb = f(xa), f(xb)
        d = (xa - xb).norm() + 1e-8
        ratio = (ya - yb).norm() / d
        (-ratio).backward()
        opt.step()

    with torch.no_grad():
        ya, yb = f(xa), f(xb)
        refined_ratio = float(((ya - yb).norm() / ((xa - xb).norm() + 1e-8)).item())

    return max(best_ratio, refined_ratio)


def check_representation_collapse(f, X, min_abs_std=1e-4, min_relative_std=1e-3):
    """
    Diagnostic: has f collapsed distinct inputs to (nearly) the same
    output vector, i.e. is it behaving like a near-constant function
    instead of a genuine, input-dependent feature map?

    When this happens, every training point looks almost identical to
    the SVM stage, so it typically degenerates to always predicting the
    majority class -- the tell-tale sign is an IDENTICAL error rate
    across many different random train/test splits, instead of the
    normal run-to-run variation.

    Parameters
    ----------
    f : callable, torch.Tensor (m, in_dim) -> torch.Tensor (m, out_dim)
    X : ndarray (m, in_dim) -- points to probe, features as ROWS (batch-first)
    min_abs_std : float
        Flag collapse if the average per-output-dimension std across the
        m points falls below this (near-constant, near-zero output).
    min_relative_std : float
        Flag collapse if that std, divided by the average output
        magnitude, falls below this (catches collapse to a large but
        still near-constant vector -- not just collapse to near-zero).

    Returns
    -------
    dict(collapsed: bool, abs_std: float, relative_std: float, mean_abs_output: float)
    """
    if not _TORCH_AVAILABLE:
        raise ImportError("torch is required for check_representation_collapse.")

    with torch.no_grad():
        Xt = torch.tensor(np.asarray(X, dtype=float), dtype=torch.float32)
        Z = f(Xt)
        Z_np = Z.detach().numpy()

    abs_std = float(Z_np.std(axis=0).mean())
    mean_abs_output = float(np.abs(Z_np).mean()) + 1e-12
    relative_std = abs_std / mean_abs_output

    collapsed = (abs_std < min_abs_std) or (relative_std < min_relative_std)
    return dict(collapsed=collapsed, abs_std=abs_std, relative_std=relative_std,
                mean_abs_output=mean_abs_output)