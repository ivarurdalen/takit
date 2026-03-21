from __future__ import annotations

import narwhals as nw
from pydantic import PositiveInt, validate_call

from takit.util.rolling import rolling_max, rolling_min
from takit.validation import config_dict, validate_ohlc


@nw.narwhalify(eager_only=True)
@validate_call(config=config_dict)
def ichimoku_cloud(
    ohlc: nw.DataFrame,
    tenkan_length: PositiveInt = 9,
    kijun_length: PositiveInt = 26,
    senkou_b_length: PositiveInt = 52,
    displacement: PositiveInt = 26,
) -> nw.DataFrame:
    """
    Ichimoku Cloud (Ichimoku Kinko Hyo).

    Reference:
        1. https://www.tradingview.com/support/solutions/43000589152-ichimoku-cloud/

    Equations:
        1. Tenkan-sen (Conversion) = (HH + LL) / 2 over tenkan_length
        2. Kijun-sen (Base) = (HH + LL) / 2 over kijun_length
        3. Senkou Span A (Leading A) = (Tenkan + Kijun) / 2, displaced forward
        4. Senkou Span B (Leading B) = (HH + LL) / 2 over senkou_b_length, displaced forward
        5. Chikou Span (Lagging) = close, displaced backward

    Args:
        ohlc: DataFrame with OHLC data
        tenkan_length: Tenkan-sen (conversion line) period
        kijun_length: Kijun-sen (base line) period
        senkou_b_length: Senkou Span B period
        displacement: Number of periods to displace Senkou spans forward

    Returns:
        DataFrame with Tenkan, Kijun, Senkou A, Senkou B, Chikou columns.
        Note: Senkou spans are NOT forward-displaced (to keep alignment with input).
        Apply .shift(displacement) on Senkou columns for charting.
    """
    ohlc = validate_ohlc(ohlc, min_length=senkou_b_length)

    high = ohlc.get_column("high")
    low = ohlc.get_column("low")

    # Tenkan-sen
    tenkan_high = rolling_max(high, tenkan_length, min_samples=1)
    tenkan_low = rolling_min(low, tenkan_length, min_samples=1)
    tenkan = ((tenkan_high + tenkan_low) / 2).alias("TENKAN")

    # Kijun-sen
    kijun_high = rolling_max(high, kijun_length, min_samples=1)
    kijun_low = rolling_min(low, kijun_length, min_samples=1)
    kijun = ((kijun_high + kijun_low) / 2).alias("KIJUN")

    # Senkou Span A (not displaced — user applies shift for charting)
    senkou_a = ((tenkan + kijun) / 2).alias("SENKOU_A")

    # Senkou Span B (not displaced)
    senkou_b_high = rolling_max(high, senkou_b_length, min_samples=1)
    senkou_b_low = rolling_min(low, senkou_b_length, min_samples=1)
    senkou_b = ((senkou_b_high + senkou_b_low) / 2).alias("SENKOU_B")

    # Chikou Span (close shifted back by displacement)
    chikou = ohlc.get_column("close").shift(-displacement).alias("CHIKOU")

    native_ns = nw.get_native_namespace(ohlc)
    data = {s.name: s.to_native() for s in [tenkan, kijun, senkou_a, senkou_b, chikou]}
    return nw.from_native(native_ns.DataFrame(data), eager_only=True)


ichimoku = ichimoku_cloud
