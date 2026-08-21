import numpy as np
import pandas as pd
import pytest

from unit_of_work_deterministic_binary import _apply_transform, prepare_binary_data


def test_binary_preparation_imputes_question_marks_nan_and_infinity():
    data = pd.DataFrame({
        "age": [40, "?", 60, np.inf],
        "density": [1.0, 2.0, np.nan, 4.0],
        "CLASS": [0, 1, 0, 1],
    })

    clean, counts = prepare_binary_data(data)
    assert counts == {"age": 2, "density": 1}
    assert np.isfinite(clean.to_numpy(dtype=float)).all()
    assert clean.loc[1, "age"] == pytest.approx(50.0)
    assert clean.loc[3, "age"] == pytest.approx(50.0)

    standardized = _apply_transform(data, "standardization")
    assert np.isfinite(standardized.to_numpy(dtype=float)).all()
    assert standardized.loc[1, "age"] == pytest.approx(0.0)


def test_binary_preparation_rejects_invalid_labels():
    data = pd.DataFrame({"feature": [1, 2, 3], "CLASS": [0, "?", 1]})
    with pytest.raises(ValueError, match="Label column"):
        prepare_binary_data(data)


def test_binary_preparation_rejects_all_missing_feature():
    data = pd.DataFrame({"feature": ["?", np.nan], "CLASS": [0, 1]})
    with pytest.raises(ValueError, match="no usable numeric values"):
        prepare_binary_data(data)
