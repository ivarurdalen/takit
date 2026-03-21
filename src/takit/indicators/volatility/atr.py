import narwhals as nw
from pydantic import PositiveInt, validate_call

from takit.indicators.trend.ma import simple_moving_average
from takit.validation import config_dict, validate_ohlc


@nw.narwhalify(eager_only=True)
@validate_call(config=config_dict)
def average_true_range(ohlc: nw.DataFrame, length: PositiveInt = 14) -> nw.Series:
    """
    Average True Range (ATR).

    Reference:
        1. https://www.tradingview.com/support/solutions/43000501823-average-true-range-atr/

    Equations:
        1. TR_i = True Range = max(high_i - low_i, abs(high_i - close_(i-1)), abs(low_i - close_(i-1)))
        2. ATR = SMA(TR, length)

    Args:
        ohlc: DataFrame with OHLC data
        length: Length of the window

    Returns:
        ATR values of the input series
    """
    ohlc = validate_ohlc(ohlc, min_length=length)

    result = ohlc.with_columns(
        (nw.col("high") - nw.col("low")).alias("high_low"),
        (nw.col("high") - nw.col("close").shift(1)).abs().alias("high_prev_close"),
        (nw.col("low") - nw.col("close").shift(1)).abs().alias("low_prev_close"),
    )
    result = result.with_columns(
        nw.max_horizontal("high_low", "high_prev_close", "low_prev_close").alias("true_range"),
    )

    true_range = result.get_column("true_range")
    sma_result = nw.from_native(simple_moving_average(true_range, length), series_only=True)
    return sma_result.alias(f"ATR{length}")


atr = average_true_range
