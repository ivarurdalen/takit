from __future__ import annotations

import narwhals as nw
from pydantic import PositiveInt, validate_call

from takit.util.rolling import rolling_max, rolling_min
from takit.validation import config_dict, validate_ohlc


@nw.narwhalify(eager_only=True)
@validate_call(config=config_dict)
def donchian_channel(ohlc: nw.DataFrame, length: PositiveInt = 20) -> nw.DataFrame:
    """
    Donchian Channel.

    Reference:
        1. https://www.tradingview.com/support/solutions/43000502253-donchian-channels-dc/

    Equations:
        1. DCH = Lookback High = max_over_range(high, length)
        2. DCL = Lookback Low = min_over_range(low, length)
        3. DCM = Lookback Mid = (DCH + DCL) / 2

    Args:
        ohlc: DataFrame with OHLC data
        length: Length of the lookback period

    Returns:
        DataFrame with DCM, DCL, DCH columns
    """
    ohlc = validate_ohlc(ohlc, min_length=length)

    dc_high = rolling_max(ohlc.get_column("high"), length, min_samples=1).alias(f"DCH{length}")
    dc_low = rolling_min(ohlc.get_column("low"), length, min_samples=1).alias(f"DCL{length}")
    dc_mid = ((dc_high + dc_low) / 2).alias(f"DCM{length}")

    native_ns = nw.get_native_namespace(ohlc)
    data = {s.name: s.to_native() for s in [dc_mid, dc_low, dc_high]}
    return nw.from_native(native_ns.DataFrame(data), eager_only=True)


dc = donchian_channel
