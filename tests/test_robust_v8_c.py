import numpy as np
import pytest

pytest.importorskip("cvxpy")

from ddr_mksvm.robust_v8_c import predict_pair, solve_pair


def test_conic_pair_is_feasible_and_penalty_grows_with_rho():
    X = np.asarray([[-2.,0.],[-1.,.1],[1.,-.1],[2.,0.]])
    y = np.asarray([0,0,1,1])
    base, d0 = solve_pair(X,y,C=1.,alpha=1.,rho=0.,p=2.)
    robust, d1 = solve_pair(X,y,C=1.,alpha=1.,rho=.05,p=2.)
    assert d0["max_constraint_violation"] < 1e-6
    assert d1["max_constraint_violation"] < 1e-6
    assert d1["max_delta"] > d0["max_delta"] == 0
    assert np.array_equal(predict_pair(base,X)>=0, np.asarray([False,False,True,True]))
    assert d1["objective_value"] + 1e-6 >= d0["objective_value"]


def test_p1_and_p2_have_same_conservative_rbf_problem():
    X=np.asarray([[-1.,0.],[-.5,.2],[.5,-.2],[1.,0.]])
    y=np.asarray([0,0,1,1])
    _, one=solve_pair(X,y,1.,1.,.02,1.)
    _, two=solve_pair(X,y,1.,1.,.02,2.)
    assert one["max_delta"] == pytest.approx(two["max_delta"])
    assert one["objective_value"] == pytest.approx(two["objective_value"], rel=1e-6)


def test_exact_duplicate_aggregation_preserves_objective_and_predictions():
    X = np.asarray([[-1., 0.], [-1., 0.], [-1., 0.], [1., 0.], [1., 0.], [1., 0.]])
    y = np.asarray([0, 0, 1, 0, 1, 1])
    sample, sample_diag = solve_pair(
        X, y, C=1., alpha=1., rho=.01, p=2., representation_mode="sample"
    )
    reduced, reduced_diag = solve_pair(
        X, y, C=1., alpha=1., rho=.01, p=2., representation_mode="deduplicate_exact"
    )
    assert reduced_diag["unique_feature_locations"] == 2
    assert reduced_diag["constraint_group_count"] == 4
    assert reduced_diag["objective_value"] == pytest.approx(sample_diag["objective_value"], rel=1e-5)
    assert np.array_equal(predict_pair(sample, X) >= 0, predict_pair(reduced, X) >= 0)


def test_sharded_prepare_records_exact_task_count_and_refuses_early_finalize(tmp_path, monkeypatch):
    from archive.research_scripts.run_robust_v8_c import parser, run
    monkeypatch.setenv("ROBUST_ALLOW_LOCAL_FIT", "1")
    args = parser().parse_args([
        "--datasets", "parkinson", "--models", "v8_c,robust_v8_c_p2",
        "--rhos", "1e-4,1e-3", "--n-seeds", "2", "--seed-start", "41",
        "--run-name", "shard-contract", "--output-root", str(tmp_path),
        "--mode", "prepare",
    ])
    run(args)
    manifest = __import__("json").loads((tmp_path / "shard-contract" / "manifest.json").read_text())
    assert manifest["expected_jobs"] == 6
    args.mode = "finalize"
    with pytest.raises(RuntimeError, match="0/6 checkpoints"):
        run(args)
