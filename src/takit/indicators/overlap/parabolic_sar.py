from __future__ import annotations

import narwhals as nw
from pydantic import validate_call

from takit.validation import config_dict, validate_ohlc


@nw.narwhalify(eager_only=True)
@validate_call(config=config_dict)
def parabolic_stop_and_reverse(
    ohlc: nw.DataFrame,
    initial_af: float = 0.02,
    af_step: float = 0.02,
    max_af: float = 0.20,
) -> nw.DataFrame:
    """
    Parabolic SAR (Stop and Reverse).

    Reference:
        1. https://www.tradingview.com/support/solutions/43000502286-parabolic-sar/
        2. J. Welles Wilder Jr., "New Concepts in Technical Trading Systems"

    Args:
        ohlc: DataFrame with OHLC data
        initial_af: Initial acceleration factor
        af_step: Acceleration factor increment
        max_af: Maximum acceleration factor

    Returns:
        DataFrame with PSAR and PSAR_TREND (1=bullish, -1=bearish) columns
    """
    ohlc = validate_ohlc(ohlc, min_length=2)

    high_list = ohlc.get_column("high").to_list()
    low_list = ohlc.get_column("low").to_list()
    close_list = ohlc.get_column("close").to_list()
    n = len(high_list)

    psar = [None] * n
    trend = [None] * n

    # Initialize: assume bullish trend starting at index 1
    is_bullish = close_list[1] >= close_list[0] if close_list[0] is not None and close_list[1] is not None else True
    af = initial_af
    if is_bullish:
        ep = high_list[0]  # Extreme point
        sar = low_list[0]
    else:
        ep = low_list[0]
        sar = high_list[0]

    psar[0] = sar
    trend[0] = 1 if is_bullish else -1

    for i in range(1, n):
        h, lo = high_list[i], low_list[i]
        if h is None or lo is None:
            psar[i] = psar[i - 1]
            trend[i] = trend[i - 1]
            continue

        prev_sar = sar
        sar = prev_sar + af * (ep - prev_sar)

        if is_bullish:
            # Clamp SAR to be below the prior two lows
            if i >= 2 and low_list[i - 1] is not None:
                sar = min(sar, low_list[i - 1])
            if i >= 2 and low_list[i - 2] is not None:
                sar = min(sar, low_list[i - 2])

            if lo < sar:
                # Reversal to bearish
                is_bullish = False
                sar = ep
                ep = lo
                af = initial_af
            else:
                if h > ep:
                    ep = h
                    af = min(af + af_step, max_af)
        else:
            # Clamp SAR to be above the prior two highs
            if i >= 2 and high_list[i - 1] is not None:
                sar = max(sar, high_list[i - 1])
            if i >= 2 and high_list[i - 2] is not None:
                sar = max(sar, high_list[i - 2])

            if h > sar:
                # Reversal to bullish
                is_bullish = True
                sar = ep
                ep = h
                af = initial_af
            else:
                if lo < ep:
                    ep = lo
                    af = min(af + af_step, max_af)

        psar[i] = sar
        trend[i] = 1 if is_bullish else -1

    native_ns = nw.get_native_namespace(ohlc)
    psar_series = nw.new_series("PSAR", psar, backend=native_ns)
    trend_series = nw.new_series("PSAR_TREND", trend, backend=native_ns)
    return nw.from_native(
        native_ns.DataFrame({"PSAR": psar_series.to_native(), "PSAR_TREND": trend_series.to_native()}),
        eager_only=True,
    )


psar = parabolic_stop_and_reverse
