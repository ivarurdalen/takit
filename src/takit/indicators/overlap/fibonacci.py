import narwhals as nw
from pydantic import validate_call

from takit.validation import config_dict

FIBONACCI_LEVELS = (0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0)


@nw.narwhalify(eager_only=True)
@validate_call(config=config_dict)
def fibonacci_retracement(
    ohlc: nw.DataFrame,
    levels: tuple[float, ...] = FIBONACCI_LEVELS,
) -> nw.DataFrame:
    """
    Fibonacci Retracement Levels.

    Computes retracement levels between the high and low of the given data.

    Reference:
        1. https://www.tradingview.com/support/solutions/43000589618-fib-retracement/

    Equations:
        Level = High - (High - Low) * ratio

    Args:
        ohlc: DataFrame with OHLC data
        levels: Fibonacci ratio levels to compute

    Returns:
        DataFrame with one column per level (e.g. FIB_0.0, FIB_0.236, ...)
        Each column is a constant value representing that retracement level.
    """
    high_val = ohlc.get_column("high").max()
    low_val = ohlc.get_column("low").min()
    diff = high_val - low_val

    native_ns = nw.get_native_namespace(ohlc)
    n = ohlc.shape[0]
    data = {}
    for level in levels:
        value = high_val - diff * level
        data[f"FIB_{level}"] = nw.new_series(
            f"FIB_{level}", [value] * n, backend=native_ns
        ).to_native()

    return nw.from_native(native_ns.DataFrame(data), eager_only=True)


fib = fibonacci_retracement
