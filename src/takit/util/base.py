from __future__ import annotations

import narwhals as nw


def value_streak(series: nw.Series, *, negative_streaks: bool = False) -> nw.Series:
    """
    Value streak.

    Calculates the cumulative count of consecutive equal values.
    For example, [1, 1, 2, 2, 2, 3] -> [1, 2, 1, 2, 3, 1]

    Args:
        series: Input series
        negative_streaks: Whether make streaks of negative values negative

    Returns:
        Streak values of the input series
    """
    values = series.to_list()
    streaks = []
    count = 0
    prev = None
    for v in values:
        if v == prev:
            count += 1
        else:
            count = 1
            prev = v
        if negative_streaks and v is not None and v < 0:
            streaks.append(-count)
        else:
            streaks.append(count)

    return nw.new_series(series.name, streaks, backend=nw.get_native_namespace(series))
