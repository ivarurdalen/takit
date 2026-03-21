from __future__ import annotations

import narwhals as nw
from pydantic import PositiveInt, validate_call

from takit.validation import config_dict, validate_series


@nw.narwhalify(eager_only=True)
@validate_call(config=config_dict)
def relative_strength_index(series: nw.Series, length: PositiveInt = 14) -> nw.Series:
    """
    Relative Strength Index (RSI).

    References:
        1. https://www.tradingview.com/support/solutions/43000502338-relative-strength-index-rsi/
        2. https://en.wikipedia.org/wiki/Relative_strength_index

    Equations:
        1. RSI = 100 - 100 / (1 + RS)
        2. RS = Relative Strength = Relative Moving Average (RMA) of gains / Relative Moving Average (RMA) of losses

    """
    series = validate_series(series, min_length=length)

    change = series.diff()

    gain = change.clip(lower_bound=0)
    loss = change.clip(upper_bound=0).abs()

    # Using Exponential Weighted Moving Average (ewm), where alpha is the smoothing factor.
    alpha = 1 / length
    avg_gain = gain.ewm_mean(alpha=alpha, min_samples=length, adjust=True)
    avg_loss = loss.ewm_mean(alpha=alpha, min_samples=length, adjust=True)

    relative_strength = avg_gain / avg_loss
    rsi_values = 100 - 100 / (1 + relative_strength)

    return rsi_values.alias(f"RSI{length}")


rsi = relative_strength_index
