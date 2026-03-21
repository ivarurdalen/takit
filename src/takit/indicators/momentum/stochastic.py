import narwhals as nw
from pydantic import PositiveInt, validate_call

from takit.indicators.trend.ma import simple_moving_average
from takit.util.rolling import rolling_max, rolling_min
from takit.validation import config_dict, validate_ohlc


@nw.narwhalify(eager_only=True)
@validate_call(config=config_dict)
def stochastic_oscillator(
    ohlc: nw.DataFrame,
    k_length: PositiveInt = 14,
    k_smoothing: PositiveInt = 1,
    d_smoothing: PositiveInt = 3,
) -> nw.DataFrame:
    """
    Stochastic Oscillator.

    Reference:
        1. https://www.tradingview.com/support/solutions/43000502332-stochastic-stoch/

    Equations:
        1. %K = 100 * (close - LL) / (HH - LL)
        2. Fast %K = SMA(%K, k_smoothing)
        3. %D = SMA(Fast %K, d_smoothing)

    Where:
        LL = Lowest low over k_length periods
        HH = Highest high over k_length periods

    Args:
        ohlc: DataFrame with OHLC data
        k_length: Lookback period for %K
        k_smoothing: Smoothing period for %K (1 = fast stochastic)
        d_smoothing: Smoothing period for %D

    Returns:
        DataFrame with %K and %D columns
    """
    ohlc = validate_ohlc(ohlc, min_length=k_length)

    close = ohlc.get_column("close")
    highest_high = rolling_max(ohlc.get_column("high"), k_length, min_samples=1)
    lowest_low = rolling_min(ohlc.get_column("low"), k_length, min_samples=1)

    raw_k = ((close - lowest_low) / (highest_high - lowest_low) * 100).alias("%K")

    if k_smoothing > 1:
        k = nw.from_native(simple_moving_average(raw_k, k_smoothing, validate=False), series_only=True).alias("%K")
    else:
        k = raw_k

    d = nw.from_native(simple_moving_average(k, d_smoothing, validate=False), series_only=True).alias("%D")

    native_ns = nw.get_native_namespace(ohlc)
    data = {s.name: s.to_native() for s in [k, d]}
    return nw.from_native(native_ns.DataFrame(data), eager_only=True)


stoch = stochastic_oscillator
