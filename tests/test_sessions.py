import pandas as pd
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from takehome.sessions import is_tradable, availability_by_week


def times(date, values):
    return pd.DatetimeIndex([f'{date} {v}' for v in values],tz='America/New_York')


def test_exchange_instants_and_bar_end_labels_are_different():
    index=times('2020-02-10',['16:10','16:15','16:20','16:30','16:35','17:00','17:05','18:00','18:05'])
    assert_array_equal(is_tradable(index),[1,0,0,1,1,0,0,1,1])
    assert_array_equal(is_tradable(index,bar_end=True),[1,1,0,0,1,1,0,0,1])


def test_weekend_and_daylight_saving_time():
    index=pd.DatetimeIndex(['2020-03-06 18:00','2020-03-07 12:00','2020-03-08 17:55',
                            '2020-03-08 18:00','2020-03-08 18:05'],tz='America/New_York')
    assert_array_equal(is_tradable(index),[0,0,0,1,1])
    assert_array_equal(is_tradable(index),is_tradable(index.tz_convert('UTC')))


def test_legacy_close_change_is_not_applied_to_holidays():
    assert is_tradable(times('2014-02-10',['17:10'])).iloc[0]
    assert not is_tradable(times('2016-02-10',['17:10'])).iloc[0]
    assert not is_tradable(times('2014-12-24',['17:10'])).iloc[0]


def test_christmas_and_juneteenth_early_close():
    assert not is_tradable(times('2020-12-25',['10:00','14:00'])).any()
    index=times('2022-06-20',['12:55','13:00','13:05','18:00','18:05'])
    assert_array_equal(is_tradable(index),[1,0,0,1,1])
    assert_array_equal(is_tradable(index,bar_end=True),[1,1,0,0,1])


def test_missing_quote_is_not_an_exchange_closure():
    index=pd.date_range('2020-02-10 09:00',periods=10,freq='5min',tz='America/New_York')
    quotes=pd.Series(1.,index=index).drop(index[2])
    quotes.iloc[3]=np.nan
    p=availability_by_week(quotes)
    assert p.loc[(0,550),'has_row']==0
    assert p.loc[(0,550),'has_wmid']==0
    assert is_tradable(index).all()


def test_timezone_is_required():
    with pytest.raises(ValueError): is_tradable(pd.date_range('2020-01-01',periods=2))


def test_afternoon_pause_removed_june_28_2021():
    index=pd.DatetimeIndex(['2021-06-25 16:20','2021-06-28 16:20'],tz='America/New_York')
    assert is_tradable(index).tolist()==[False,True]
    assert is_tradable(index,bar_end=True).tolist()==[False,True]
