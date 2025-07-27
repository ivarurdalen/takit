import pandas as pd
from pydantic import PositiveInt, validate_call

from takit.validation import config_dict, ta_series_schema


@validate_call(config=config_dict)
def relative_strength_index(series: pd.Series, length: PositiveInt = 14) -> pd.Series:
    """
    Relative Strength Index (RSI).

    References:
        1. https://www.tradingview.com/support/solutions/43000502338-relative-strength-index-rsi/
        2. https://en.wikipedia.org/wiki/Relative_strength_index

    Equations:
        1. RSI = 100 - 100 / (1 + RS)
        2. RS = Relative Strength = Relative Moving Average (RMA) of gains / Relative Moving Average (RMA) of losses

    """
    series = ta_series_schema(min_length=length).validate(series)

    df = series.to_frame(name="series").copy()

    df["change"] = df["series"].diff()
    df = df.dropna(axis=0, how="any")
    df["gain"] = df["change"].clip(lower=0)
    df["loss"] = df["change"].clip(upper=0).abs()

    # Using Exponential Weighted Moving Average (ewm), where alpha is the smoothing factor.
    alpha = 1 / length
    df["avg_gain"] = df["gain"].ewm(alpha=alpha, min_periods=length).mean()
    df["avg_loss"] = df["loss"].ewm(alpha=alpha, min_periods=length).mean()

    df["relative_strength"] = df["avg_gain"] / df["avg_loss"]
    df[f"RSI{length}"] = 100 - 100 / (1 + df["relative_strength"])

    return df[f"RSI{length}"]


rsi = relative_strength_index
