import narwhals as nw
from pydantic import PositiveInt, validate_call

from takit.indicators.trend.ma import relative_moving_average
from takit.indicators.volatility.atr import average_true_range
from takit.validation import config_dict, validate_ohlc


@nw.narwhalify(eager_only=True)
@validate_call(config=config_dict)
def average_directional_index(ohlc: nw.DataFrame, length: PositiveInt = 14) -> nw.DataFrame:
    """
    Average Directional Index (ADX).

    Reference:
        1. https://www.tradingview.com/support/solutions/43000502250-average-directional-index-adx/

    Equations:
        1. +DM = high - prev_high (if > 0 and > -(low - prev_low), else 0)
        2. -DM = prev_low - low (if > 0 and > high - prev_high, else 0)
        3. +DI = 100 * RMA(+DM, length) / ATR
        4. -DI = 100 * RMA(-DM, length) / ATR
        5. DX = 100 * abs(+DI - -DI) / (+DI + -DI)
        6. ADX = RMA(DX, length)

    Args:
        ohlc: DataFrame with OHLC data
        length: Period for smoothing

    Returns:
        DataFrame with +DI, -DI, and ADX columns
    """
    ohlc = validate_ohlc(ohlc, min_length=length)

    high = ohlc.get_column("high")
    low = ohlc.get_column("low")

    # Directional movement
    high_diff = high.diff()
    low_diff = low.diff() * -1  # prev_low - low

    high_list = high_diff.to_list()
    low_list = low_diff.to_list()

    plus_dm_vals = []
    minus_dm_vals = []
    for hd, ld in zip(high_list, low_list, strict=True):
        if hd is None or ld is None:
            plus_dm_vals.append(None)
            minus_dm_vals.append(None)
        elif hd > ld and hd > 0:
            plus_dm_vals.append(hd)
            minus_dm_vals.append(0.0)
        elif ld > hd and ld > 0:
            plus_dm_vals.append(0.0)
            minus_dm_vals.append(ld)
        else:
            plus_dm_vals.append(0.0)
            minus_dm_vals.append(0.0)

    native_ns = nw.get_native_namespace(ohlc)
    plus_dm = nw.new_series("+DM", plus_dm_vals, backend=native_ns)
    minus_dm = nw.new_series("-DM", minus_dm_vals, backend=native_ns)

    # Smoothed DM and ATR
    smoothed_plus_dm = nw.from_native(relative_moving_average(plus_dm, length), series_only=True)
    smoothed_minus_dm = nw.from_native(relative_moving_average(minus_dm, length), series_only=True)
    atr_values = nw.from_native(average_true_range(ohlc, length), series_only=True)

    # Directional indicators
    plus_di = (100 * smoothed_plus_dm / atr_values).alias(f"+DI{length}")
    minus_di = (100 * smoothed_minus_dm / atr_values).alias(f"-DI{length}")

    # ADX
    di_sum = plus_di + minus_di
    di_diff = (plus_di - minus_di).abs()

    # Avoid division by zero: compute DX manually
    di_sum_list = di_sum.to_list()
    di_diff_list = di_diff.to_list()
    dx_vals = [
        100 * d / s if s is not None and s != 0 and d is not None else None
        for d, s in zip(di_diff_list, di_sum_list, strict=True)
    ]
    dx = nw.new_series("DX", dx_vals, backend=native_ns)
    adx = nw.from_native(relative_moving_average(dx, length), series_only=True).alias(f"ADX{length}")

    data = {s.name: s.to_native() for s in [plus_di, minus_di, adx]}
    return nw.from_native(native_ns.DataFrame(data), eager_only=True)


adx = average_directional_index
