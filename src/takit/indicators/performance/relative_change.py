import pandas as pd
from pydantic import validate_call

from takit.validation import config_dict, ta_series_schema


@validate_call(config=config_dict)
def relative_change(series: pd.Series, start: pd.Timestamp | None = None) -> pd.Series:
    """
    Relative Change (RC).

    Calculate the relative change of a series from a start date.

    Args:
        series: Input series with DatetimeIndex
        start: Start date. If None, use the first date in the series.
    """
    series = ta_series_schema(min_length=2).validate(series)

    if start is not None:
        series = series.loc[start:]

    return (series / series.iloc[0] - 1).rename("RC")
