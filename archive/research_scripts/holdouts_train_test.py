"""
Python translation of holdouts_train_test.m

Splits DATA (a pandas DataFrame whose last column is the binary class
label, coded as 1 / 0) into a stratified train/test holdout, then
further separates each split into the two classes (A = class 1,
B = class 0), returning plain numpy arrays with the class column
already stripped off - exactly like the original MATLAB function.
"""

import numpy as np
from sklearn.model_selection import train_test_split


def holdouts_train_test(DATA, testingsamplesize, random_state=None):
    """
    Parameters
    ----------
    DATA : pandas.DataFrame
        Last column must be the class label (1 or 0).
    testingsamplesize : float
        Fraction of the data to hold out for testing (e.g. 0.25).
    random_state : int or None
        Passed straight to sklearn's train_test_split. Default None
        preserves the original behavior (a fresh, uncontrolled random
        split every call -- the 96-independent-runs protocol). Pass an
        explicit int (e.g. the run index) to make a given call's split
        reproducible -- needed for PAIRED comparisons across scripts/
        ablations that should see the exact same train/test partition.

    Returns
    -------
    Xtrain : ndarray  -- class-1 training samples (features only)
    Xtest  : ndarray  -- class-1 testing samples  (features only)
    Ytrain : ndarray  -- class-0 training samples (features only)
    Ytest  : ndarray  -- class-0 testing samples  (features only)
    """

    class_col = DATA.columns[-1]

    idx_train, idx_test = train_test_split(
        DATA.index,
        test_size=testingsamplesize,
        stratify=DATA[class_col],
        random_state=random_state,
    )

    DATAtrain = DATA.loc[idx_train]
    DATAtest = DATA.loc[idx_test]

    # class 1 -> "X", class 0 -> "Y"  (mirrors the MATLAB code)
    X = DATAtrain[DATAtrain[class_col] == 1].copy()
    Y = DATAtrain[DATAtrain[class_col] == 0].copy()

    Xtest = DATAtest[DATAtest[class_col] == 1].copy()
    Ytest = DATAtest[DATAtest[class_col] == 0].copy()

    # drop the class column
    X = X.drop(columns=[class_col])
    Y = Y.drop(columns=[class_col])
    Xtest = Xtest.drop(columns=[class_col])
    Ytest = Ytest.drop(columns=[class_col])

    Xtrain = X.to_numpy(dtype=float)
    Ytrain = Y.to_numpy(dtype=float)
    Xtest = Xtest.to_numpy(dtype=float)
    Ytest = Ytest.to_numpy(dtype=float)

    return Xtrain, Xtest, Ytrain, Ytest
