"""
Python translation of holdouts_train_test_multiclass.m

Stratified train/test holdout split of a multiclass DataFrame whose
class column is named 'CLASS' (integer labels 1..L, matching the
MATLAB convention). Mirrors cvpartition(...,'HoldOut',...) + table2array:
both outputs keep the class column as the LAST column of a plain
numpy array, exactly like the original MATLAB function.
"""

import numpy as np
from sklearn.model_selection import train_test_split


def holdouts_train_test_multiclass(DATA, testingsamplesize, random_state=None):
    """
    Parameters
    ----------
    DATA : pandas.DataFrame
        Must contain a 'CLASS' column (multiclass integer labels, 1..L).
        The 'CLASS' column does not need to be the last column in DATA;
        it is moved there to match the MATLAB table2array layout.
    testingsamplesize : float
        Fraction of the data to hold out for testing (e.g. 0.25).
    random_state : int or None
        See holdouts_train_test.py's docstring -- same semantics.

    Returns
    -------
    DATAtrain : ndarray -- training rows, class label in the last column
    DATAtest  : ndarray -- testing rows,  class label in the last column
    """

    # make sure CLASS ends up as the last column, like table2array(DATA)
    # would if CLASS were the last variable in the MATLAB table
    cols = [c for c in DATA.columns if c != 'CLASS'] + ['CLASS']
    DATA = DATA[cols]

    idx_train, idx_test = train_test_split(
        DATA.index,
        test_size=testingsamplesize,
        stratify=DATA['CLASS'],
        random_state=random_state,
    )

    DATAtrain = DATA.loc[idx_train].to_numpy(dtype=float)
    DATAtest = DATA.loc[idx_test].to_numpy(dtype=float)

    return DATAtrain, DATAtest
