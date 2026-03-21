from typing import Annotated

import narwhals as nw
from pydantic import ConfigDict, StringConstraints

PriceCol = Annotated[str, StringConstraints(pattern="close|high|low|open")]

config_dict = ConfigDict(arbitrary_types_allowed=True)

OHLC_COLUMNS = {"open", "high", "low", "close"}


def validate_ohlc(df: nw.DataFrame, min_length: int = 1) -> nw.DataFrame:
    """
    Validate OHLC DataFrame.

    Args:
        df: Input DataFrame (narwhals-wrapped)
        min_length: Minimum number of rows required

    Raises:
        ValueError: If validation fails
    """
    missing = OHLC_COLUMNS - set(df.columns)
    if missing:
        msg = f"Missing OHLC columns: {missing}"
        raise ValueError(msg)
    if df.shape[0] <= min_length:
        msg = f"DataFrame must have more than {min_length} rows."
        raise ValueError(msg)
    return df


def validate_series(series: nw.Series, min_length: int = 1) -> nw.Series:
    """
    Validate a Series for technical analysis.

    Args:
        series: Input Series (narwhals-wrapped)
        min_length: Minimum number of values required

    Raises:
        ValueError: If validation fails
    """
    if series.len() <= min_length:
        msg = f"Series must have more than {min_length} values."
        raise ValueError(msg)
    return series
