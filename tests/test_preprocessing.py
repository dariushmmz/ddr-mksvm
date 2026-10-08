import numpy as np

from ddr_mksvm.preprocessing import fit_transform_train_test


def test_scaler_is_fitted_on_training_rows_only():
    train = np.array([[0.0], [2.0]])
    test = np.array([[100.0]])
    train_t, test_t = fit_transform_train_test(train, test, "standardization")
    assert np.allclose(train_t[:, 0], [-1.0, 1.0])
    assert np.allclose(test_t[:, 0], [99.0])


def test_minmax_does_not_clip_unseen_test_extremes():
    train_t, test_t = fit_transform_train_test(
        np.array([[1.0], [3.0]]), np.array([[5.0]]), "minmax")
    assert np.allclose(train_t[:, 0], [0.0, 1.0])
    assert np.allclose(test_t[:, 0], [2.0])
