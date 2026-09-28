"""Scheduled ES tradability, separate from observed quote availability.

CME schedule/holidays: https://www.cmegroup.com/trading-hours.html
Historical close change: https://www.cmegroup.com/tools-information/lookups/advisories/electronic-trading/20150817.html
Halt removal: https://www.cmegroup.com/notices/electronic-trading/2021/06/20210621.html
Mourning session: https://www.cmegroup.com/notices/clearing/2018/12/Chadv18-474.html
Calendar rules are supplied by pandas_market_calendars, not a live halt feed.
"""
import numpy as np
import pandas as pd


def _index(index):
    index = pd.DatetimeIndex(index)
    if index.tz is None or index.hasnans or not index.is_unique or not index.is_monotonic_increasing:
        raise ValueError('Require sorted, unique, timezone-aware timestamps without NaT.')
    return index


def es_schedule(index):
    """CME equity calendar plus dated ES corrections; not unscheduled halts."""
    import pandas_market_calendars as mcal

    index = _index(index)
    if not len(index):
        raise ValueError('Cannot build a schedule for an empty index.')
    local = index.tz_convert('America/New_York')
    calendar = mcal.get_calendar('CME_Equity')
    schedule = calendar.schedule((local[0]-pd.Timedelta(days=3)).date(),
                                 (local[-1]+pd.Timedelta(days=3)).date())
    # Official CME notice: regular close changed from 16:15 CT to 16:00 CT.
    close = schedule.market_close.dt.tz_convert('America/Chicago')
    legacy = ((schedule.index < '2015-09-21') & close.dt.hour.eq(16) & close.dt.minute.eq(0))
    schedule.loc[legacy, 'market_close'] += pd.Timedelta(minutes=15)
    # ES afternoon pause was removed from trade date 2021-06-28.
    modern = schedule.index >= '2021-06-28'
    for field in ['break_start', 'break_end']:
        schedule.loc[modern, field] = schedule.loc[modern, 'market_close']
    # Unlike the cash market, ES had an overnight session on this mourning day.
    day = pd.Timestamp('2018-12-05')
    if schedule.index.min() <= day <= schedule.index.max():
        opening = pd.Timestamp('2018-12-04 18:00', tz='America/New_York').tz_convert('UTC')
        closing = pd.Timestamp('2018-12-05 09:30', tz='America/New_York').tz_convert('UTC')
        schedule.loc[day] = [opening if col == 'market_open' else closing for col in schedule]
    return schedule.sort_index()


def is_tradable(index, *, bar_end=False, bar_size='5min'):
    """Instantaneous session-open mask; bar_end=True tests a complete trailing bar.

    Instant: [open, close), excluding [break_start, break_end).
    Bar: its whole (t-bar_size, t] lies in an open interval; close labels count.
    No contemporaneous/future wmid or return is required to generate this mask.
    """
    index = _index(index)
    if not len(index):
        return pd.Series([], index=index, dtype=bool, name='is_tradable')
    schedule = es_schedule(index)
    if {'break_start', 'break_end'} <= set(schedule):
        starts = pd.concat([schedule.market_open, schedule.break_end])
        stops = pd.concat([schedule.break_start, schedule.market_close])
    else:
        starts, stops = schedule.market_open, schedule.market_close
    valid = (stops > starts).to_numpy()
    left = pd.DatetimeIndex(starts).as_unit('ns').asi8[valid]
    right = pd.DatetimeIndex(stops).as_unit('ns').asi8[valid]
    order = np.argsort(left)
    left, right = left[order], right[order]
    times = index.as_unit('ns').asi8
    width = pd.Timedelta(bar_size).value if bar_end else 0
    if width < 0 or (bar_end and width == 0):
        raise ValueError('bar_size must be positive.')
    positions = np.searchsorted(left, times-width, side='right') - 1
    safe = np.maximum(positions, 0)
    result = (positions >= 0) & (times <= right[safe] if bar_end else times < right[safe])
    return pd.Series(result, index=index, name='is_tradable')


def availability_by_week(wmid, freq='5min'):
    """Training-only weekday/clock frequencies; include absent rows in the denominator."""
    index = _index(wmid.index)
    utc = index.tz_convert('UTC')
    grid = pd.date_range(utc[0].floor(freq), utc[-1].ceil(freq), freq=freq).tz_convert('America/New_York')
    availability = pd.DataFrame({
        'has_row': pd.Series(True,index=index).reindex(grid,fill_value=False),
        'has_wmid': wmid.reindex(grid).notna(),
    },index=grid)
    grouped = availability.groupby([grid.dayofweek, grid.hour*60+grid.minute])
    profile = grouped.mean()
    profile['observations'] = grouped.size()
    profile.index.names = ['weekday','minute']
    return profile
