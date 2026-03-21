import narwhals as nw
from pydantic import validate_call

from takit.validation import config_dict


@nw.narwhalify(eager_only=True)
@validate_call(config=config_dict)
def on_balance_volume(ohlc: nw.DataFrame) -> nw.Series:
    """
    On-Balance Volume (OBV).

    Reference:
        1. https://www.tradingview.com/support/solutions/43000502088-on-balance-volume-obv/

    Logic:
        If close > prev_close: OBV += volume
        If close < prev_close: OBV -= volume
        If close == prev_close: OBV unchanged

    Args:
        ohlc: DataFrame with OHLC and volume data

    Returns:
        OBV values
    """
    missing = {"close", "volume"} - set(ohlc.columns)
    if missing:
        msg = f"Missing columns: {missing}"
        raise ValueError(msg)

    close_list = ohlc.get_column("close").to_list()
    volume_list = ohlc.get_column("volume").to_list()

    obv_values = [0.0]
    for i in range(1, len(close_list)):
        c, pc, v = close_list[i], close_list[i - 1], volume_list[i]
        if c is None or pc is None or v is None:
            obv_values.append(obv_values[-1])
        elif c > pc:
            obv_values.append(obv_values[-1] + v)
        elif c < pc:
            obv_values.append(obv_values[-1] - v)
        else:
            obv_values.append(obv_values[-1])

    return nw.new_series("OBV", obv_values, backend=nw.get_native_namespace(ohlc))


obv = on_balance_volume
