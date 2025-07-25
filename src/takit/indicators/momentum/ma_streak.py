from typing import Annotated

import pandas as pd
from pydantic import PositiveInt, StringConstraints, validate_call

from takit.indicators.trend import ma
from takit.util.base import value_streak
from takit.validation import config_dict

DEFAULT_MA_LENGTH = 20


@validate_call(config=config_dict)
def moving_average_streak(
    series: pd.Series,
    length: PositiveInt = DEFAULT_MA_LENGTH,
    mode: Annotated[str, StringConstraints(pattern="sma|ema")] = "sma",
) -> pd.DataFrame:
    """
    Moving average streak.

    Args:
        series: Input series
        length: Length of the moving average window
        mode: Mode of the moving average (sma or ema)

    Returns:
        DataFrame with moving average and streak values
    """
    df = series.to_frame()

    col_name = f"{mode.upper()}{length}"
    df[col_name] = getattr(ma, mode)(series, length=length)

    change = df[col_name].diff()

    def _wrangle_change(x: float) -> int:
        return 1 if x >= 0 else -1

    change = change.map(_wrangle_change)

    df[f"{col_name}STREAK"] = value_streak(change, negative_streaks=True)

    return df.iloc[:, -2:]


ma_streak = moving_average_streak
