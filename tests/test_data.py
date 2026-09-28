import numpy as np
import pandas as pd
import pytest

from takehome.data import column_coverage, training_data


def example():
    stamps = pd.to_datetime([
        '2019-12-31', '2020-01-01', '2020-01-02', '2020-01-08',
        '2020-01-09', '2020-01-10', '2020-01-11', '2020-01-12',
    ])
    return pd.DataFrame({
        'ret_5m': [np.nan, 1., np.nan, 3., 4., 5., 6., np.nan],
        'x1': np.arange(8.), 'empty': np.nan,
    }, index=pd.DatetimeIndex(stamps, name='msgStamp'))


def test_split_uses_elapsed_label_span_not_row_count():
    df = example()
    train, split = training_data(df)
    assert split['cutoff'] == pd.Timestamp('2020-01-09')
    assert train.index.tolist() == list(pd.to_datetime(['2020-01-01', '2020-01-02', '2020-01-08']))
    assert split['train_rows'] == 3 and split['test_rows'] == 3
    assert split['first_label'] == df.ret_5m.first_valid_index()
    assert split['last_label'] == df.ret_5m.last_valid_index()


def test_split_keeps_internal_missing_labels_and_excludes_unlabeled_tails():
    train, split = training_data(example())
    assert train.ret_5m.isna().sum() == 1
    assert train.index.min() == split['first_label']
    assert train.index.max() < split['cutoff']


def test_holdout_values_do_not_change_training_data():
    original = example()
    changed = original.copy()
    changed.loc['2020-01-09':, ['ret_5m', 'x1']] *= 1000
    pd.testing.assert_frame_equal(training_data(original)[0], training_data(changed)[0])


def test_coverage_uses_valid_msgstamp_and_retains_empty_columns():
    df = example()
    out = column_coverage(df)
    assert out.loc['ret_5m', 'first_valid'] == pd.Timestamp('2020-01-01')
    assert out.loc['ret_5m', 'last_valid'] == pd.Timestamp('2020-01-11')
    assert out.loc['ret_5m', 'null_pct'] == 37.5
    assert out.loc['empty', 'null_pct'] == 100.
    assert pd.isna(out.loc['empty', 'first_valid'])
    assert pd.isna(out.loc['empty', 'last_valid'])
    assert list(out.index) == list(df.columns)


@pytest.mark.parametrize('fraction', [0, 1, -.1, 1.1])
def test_invalid_split_fraction(fraction):
    with pytest.raises(ValueError):
        training_data(example(), fraction=fraction)


def test_empty_label_span_is_rejected():
    df = example()
    df['ret_5m'] = np.nan
    with pytest.raises(ValueError, match='label'):
        training_data(df)


def test_unsorted_or_duplicate_time_index_is_rejected():
    df = example()
    for bad in [df.iloc[::-1], pd.concat([df.iloc[:1], df])]:
        with pytest.raises(ValueError, match='index'):
            training_data(bad)
