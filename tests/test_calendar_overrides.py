import pandas as pd
from takehome.sessions import is_tradable


def test_2018_mourning_session_is_shortened_not_fully_closed():
    idx = pd.DatetimeIndex(['2018-12-05 09:25', '2018-12-05 09:30',
                            '2018-12-05 10:00', '2018-12-05 18:00'], tz='America/New_York')
    assert is_tradable(idx).tolist() == [True, False, False, True]
    assert is_tradable(idx, bar_end=True).tolist() == [True, True, False, False]


def test_good_friday_2015_is_a_short_session_but_2022_is_closed():
    idx = pd.DatetimeIndex(['2015-04-03 09:10', '2015-04-03 09:20',
                            '2022-04-15 09:10'], tz='America/New_York')
    assert is_tradable(idx).tolist() == [True, False, False]
