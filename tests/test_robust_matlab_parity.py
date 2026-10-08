import math
import numpy as np
import pytest

from ddr_mksvm.robust_matlab_parity import (KernelSpec, build_q1_lp, class_eta, fit_binary, fit_ova, gram,
    matlab_polynomial_delta, p_constant, polynomial_delta_point, predict_ova,
    uncertainty_delta)


def toy():
    X = np.asarray([[-2., 0.], [-1., .2], [1., -.1], [2., 0.]])
    labels = np.asarray([0, 0, 1, 1])
    y = np.where(labels == 1, 1., -1.)
    return X, labels, y


def test_p_constants_and_dual_norm_cases():
    assert p_constant(9, 1) == 1
    assert p_constant(9, 2) == 1
    assert p_constant(9, math.inf) == 3
    assert p_constant(16, 4) == pytest.approx(16**0.25)


def test_kernel_matrices_are_symmetric_psd():
    X, _, _ = toy()
    for spec in (KernelSpec("poly", 2, 0.), KernelSpec("poly", 2, 1.), KernelSpec("rbf", alpha=1.2)):
        K = gram(X, None, spec)
        assert np.allclose(K, K.T)
        assert np.linalg.eigvalsh(K).min() > -1e-10


def test_rbf_radius_formula_and_p_monotonicity():
    X, labels, _ = toy(); spec = KernelSpec("rbf", alpha=2.)
    d1 = uncertainty_delta(X, labels, .1, 1, spec)
    d2 = uncertainty_delta(X, labels, .1, 2, spec)
    di = uncertainty_delta(X, labels, .1, math.inf, spec)
    assert np.allclose(d1, d2)
    assert np.all(di >= d2)
    assert np.all(uncertainty_delta(X, labels, 0., math.inf, spec) == 0)


def test_rho_scales_class_eta_linearly():
    X,labels,_=toy()
    first=class_eta(X,labels,.01);second=class_eta(X,labels,.02)
    assert all(second[key] == pytest.approx(2*first[key]) for key in first)


def test_active_homogeneous_polynomial_loop_matches_paper():
    X, labels, _ = toy(); rho = .03; p = math.inf
    actual = matlab_polynomial_delta(X, labels, rho, p, 2, 0.)
    etas = {c: rho*np.max(np.std(X[labels==c], axis=0, ddof=1)) for c in (0,1)}
    expected = np.asarray([polynomial_delta_point(x, etas[c], p, 2, 0.) for x,c in zip(X,labels)])
    assert np.allclose(actual, expected)


def test_inhomogeneous_source_defect_is_detectable_not_hidden():
    X, labels, _ = toy(); rho = .03; p = math.inf
    source = matlab_polynomial_delta(X, labels, rho, p, 3, 1.)
    etas = {c: rho*np.max(np.std(X[labels==c], axis=0, ddof=1)) for c in (0,1)}
    paper = np.asarray([polynomial_delta_point(x, etas[c], p, 3, 1.) for x,c in zip(X,labels)])
    assert not np.allclose(source, paper)


def test_lp_objective_and_constraint_shapes():
    X, _, y = toy(); K = gram(X, None, KernelSpec("poly", 1, 0.)); delta = np.arange(4)/100
    c, A, b, bounds = build_q1_lp(K, y, .25, delta)
    assert c.shape == (13,) and A.shape == (12, 13) and b.shape == (12,) and len(bounds) == 13
    assert np.all(c[5:9] == .25) and np.all(c[9:] == 1)
    assert np.allclose(A[:4, 9:], np.outer(delta, np.sqrt(np.diag(K))))


def test_rho_zero_deterministic_reduction_and_feasibility():
    X, labels, y = toy(); spec = KernelSpec("rbf", alpha=1.)
    first, d0 = fit_binary(X, y, labels, spec, 0., math.inf)
    second, d1 = fit_binary(X, y, labels, spec, 0., 1.)
    assert np.allclose(first.u, second.u)
    assert first.b == pytest.approx(second.b)
    assert first.diagnostics["max_constraint_violation"] < 1e-7
    assert all(row["max_delta"] == 0 for row in d0+d1)
    lo=min(first.diagnostics["threshold_left"],first.diagnostics["threshold_right"])
    hi=max(first.diagnostics["threshold_left"],first.diagnostics["threshold_right"])
    assert lo <= first.b <= hi


def test_multiclass_source_constructs_one_model_per_sorted_class():
    X=np.asarray([[-3.,0.],[-2.,.1],[-1.,-.1],[0.,0.],[.2,.1],[.1,-.1],[2.,0.],[3.,.1],[2.5,-.1]])
    labels=np.repeat([1,2,3],3)
    models,diagnostics=fit_ova(X,labels,KernelSpec("poly",1,0.),0.,math.inf)
    assert [m.positive_class for m in models] == [1,2,3]
    assert len(diagnostics)==3*5
    assert all(row["threshold_source_bug"]=="nominal-if/robust-store/delta-inside-D" for row in diagnostics)


def test_ova_tie_behavior_is_explicit():
    class Dummy:
        positive_class = 1
        def decision(self, X): return np.zeros(len(X))
    second = Dummy(); second.positive_class = 2
    with pytest.raises(RuntimeError, match="undefined"):
        predict_ova([Dummy(), second], np.zeros((2, 1)), "matlab_error")
    pred, _, ties = predict_ova([Dummy(), second], np.zeros((2, 1)), "first")
    assert np.array_equal(pred, [1,1]) and ties == 2


def test_sharded_prepare_records_exact_task_count_and_refuses_early_finalize(tmp_path, monkeypatch):
    from archive.research_scripts.run_robust_matlab_parity import parser, run
    monkeypatch.setenv("ROBUST_ALLOW_LOCAL_FIT", "1")
    args = parser().parse_args([
        "--datasets", "parkinson", "--models", "deterministic,robust_p2",
        "--rhos", "1e-4,1e-3", "--n-seeds", "2", "--seed-start", "41",
        "--run-name", "shard-contract", "--output-root", str(tmp_path),
        "--mode", "prepare",
    ])
    run(args)
    manifest = __import__("json").loads((tmp_path / "shard-contract" / "manifest.json").read_text())
    assert manifest["expected_jobs"] == 6  # one deterministic + two rho jobs, per seed
    args.mode = "finalize"
    with pytest.raises(RuntimeError, match="0/6 checkpoints"):
        run(args)
