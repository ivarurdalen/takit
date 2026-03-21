from __future__ import annotations

import narwhals as nw
from pydantic import PositiveInt, validate_call

from takit.indicators.trend.ma import exponential_moving_average
from takit.indicators.volatility.atr import average_true_range
from takit.validation import config_dict, validate_ohlc


@nw.narwhalify(eager_only=True)
@validate_call(config=config_dict)
def keltner_channel(
    ohlc: nw.DataFrame,
    ema_length: PositiveInt = 20,
    atr_length: PositiveInt = 10,
    multiplier: float = 1.5,
) -> nw.DataFrame:
    """
    Keltner Channel.

    Reference:
        1. https://www.tradingview.com/support/solutions/43000502266-keltner-channels-kc/

    Equations:
        1. Basis = EMA(close, ema_length)
        2. Upper = Basis + multiplier * ATR(atr_length)
        3. Lower = Basis - multiplier * ATR(atr_length)

    Args:
        ohlc: DataFrame with OHLC data
        ema_length: EMA period for the basis
        atr_length: ATR period
        multiplier: ATR multiplier for the bands

    Returns:
        DataFrame with KCM (middle), KCL (lower), KCU (upper) columns
    """
    ohlc = validate_ohlc(ohlc, min_length=max(ema_length, atr_length))

    close = ohlc.get_column("close")
    basis = nw.from_native(exponential_moving_average(close, ema_length), series_only=True).alias(
        f"KCM{ema_length}"
    )
    atr_values = nw.from_native(average_true_range(ohlc, atr_length), series_only=True)

    upper = (basis + multiplier * atr_values).alias(f"KCU{ema_length}_{multiplier}")
    lower = (basis - multiplier * atr_values).alias(f"KCL{ema_length}_{multiplier}")

    native_ns = nw.get_native_namespace(ohlc)
    data = {s.name: s.to_native() for s in [basis, lower, upper]}
    return nw.from_native(native_ns.DataFrame(data), eager_only=True)


kc = keltner_channel
