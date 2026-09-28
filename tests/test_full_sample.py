import numpy as np
import pandas as pd
from takehome.eda import bell_shape_table, autocorrelation_table


def test_shape_screen_does_not_subsample():
    df = pd.DataFrame({'x': np.random.default_rng(0).standard_t(3, 12000)})
    out = bell_shape_table(df, ['x']).iloc[0]
    assert out['n'] == len(df)
    assert out['nunique'] == len(df)
    assert out['bell_like']


def test_autocorrelation_uses_the_entire_input():
    df = pd.DataFrame({'x': np.random.default_rng(1).normal(size=120001)})
    assert autocorrelation_table(df, ['x']).iloc[0]['n'] == len(df)
