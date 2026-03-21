import narwhals as nw
from pydantic import PositiveInt, validate_call

from takit.validation import config_dict


@nw.narwhalify(eager_only=True)
@validate_call(config=config_dict)
def money_flow_index(ohlc: nw.DataFrame, length: PositiveInt = 14) -> nw.Series:
    """
    Money Flow Index (MFI).

    RSI applied to money flow (typical price * volume).

    Reference:
        1. https://www.tradingview.com/support/solutions/43000502245-money-flow-index-mfi/

    Equations:
        1. Typical Price = (high + low + close) / 3
        2. Raw Money Flow = Typical Price * Volume
        3. Positive MF = sum of Raw MF when TP > prev TP over length
        4. Negative MF = sum of Raw MF when TP < prev TP over length
        5. MFI = 100 - 100 / (1 + Positive MF / Negative MF)

    Args:
        ohlc: DataFrame with OHLC and volume data
        length: Lookback period

    Returns:
        MFI values
    """
    missing = {"high", "low", "close", "volume"} - set(ohlc.columns)
    if missing:
        msg = f"Missing columns: {missing}"
        raise ValueError(msg)

    high = ohlc.get_column("high").to_list()
    low = ohlc.get_column("low").to_list()
    close = ohlc.get_column("close").to_list()
    volume = ohlc.get_column("volume").to_list()

    n = len(close)
    tp = [(h + lo + c) / 3 if h is not None and lo is not None and c is not None else None
          for h, lo, c in zip(high, low, close, strict=True)]
    raw_mf = [t * v if t is not None and v is not None else None for t, v in zip(tp, volume, strict=True)]

    mfi_values = [None] * n
    for i in range(length, n):
        pos_mf = 0.0
        neg_mf = 0.0
        for j in range(i - length + 1, i + 1):
            if tp[j] is None or tp[j - 1] is None or raw_mf[j] is None:
                continue
            if tp[j] > tp[j - 1]:
                pos_mf += raw_mf[j]
            elif tp[j] < tp[j - 1]:
                neg_mf += raw_mf[j]
        if neg_mf == 0:
            mfi_values[i] = 100.0
        else:
            mfi_values[i] = 100.0 - 100.0 / (1.0 + pos_mf / neg_mf)

    return nw.new_series(f"MFI{length}", mfi_values, backend=nw.get_native_namespace(ohlc))


mfi = money_flow_index
