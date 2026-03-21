from typing import Annotated

import narwhals as nw
from pydantic import BaseModel, PositiveInt, StringConstraints, validate_call

from takit.indicators.trend import ma
from takit.validation import config_dict, validate_series


class MA(BaseModel):
    length: PositiveInt
    mode: Annotated[str, StringConstraints(pattern="sma|ema")] = "sma"
    multiplier: float = 1

    def get_col_name(self) -> str:
        """Get the column name for the moving average."""
        col = f"{self.mode.upper()}{self.length}"
        if self.multiplier != 1.0:
            col += f"x{self.multiplier}"
        return col


@nw.narwhalify(eager_only=True)
@validate_call(config=config_dict)
def ma_cross(
    series: nw.Series, fast_ma: MA, slow_ma: MA, signal_name: str | None = None, *, only_crosses: bool = False
) -> nw.DataFrame:
    """
    MA Cross.

    Reference:
        https://tradeciety.com/trend-rider-indicator

    Logic:
       If Value > Fast MA and Value > Slow MA -> Uptrend
       Else If Value < Fast MA and Value < Slow MA -> Downtrend
       Else -> Sideways / Undetermined

    Args:
        series: Input series
        fast_ma: Fast moving average
        slow_ma: Slow moving average
        signal_name: Name of the signal
        only_crosses: If True, only set 1 when fast_ma crosses slow_ma, otherwise set 1 when fast_ma > slow_ma

    Returns:
        Trend values of the input series
    """
    series = validate_series(series, min_length=slow_ma.length)

    fast_col = fast_ma.get_col_name()
    slow_col = slow_ma.get_col_name()

    fast_values = nw.from_native(
        getattr(ma, fast_ma.mode)(series, fast_ma.length), series_only=True
    )
    fast_values = (fast_values * fast_ma.multiplier).alias(fast_col)

    slow_values = nw.from_native(
        getattr(ma, slow_ma.mode)(series, slow_ma.length), series_only=True
    )
    slow_values = (slow_values * slow_ma.multiplier).alias(slow_col)

    if signal_name is None:
        signal_name = f"{fast_col}X{slow_col}"

    native_ns = nw.get_native_namespace(series)
    df = nw.from_native(
        native_ns.DataFrame({
            fast_col: fast_values.to_native(),
            slow_col: slow_values.to_native(),
        }),
        eager_only=True,
    )

    df = df.with_columns(
        nw.when(nw.col(fast_col) > nw.col(slow_col)).then(1).otherwise(-1).alias(signal_name)
    )

    if only_crosses:
        # Detect crossover points: diff of signal is 2 (cross up) or -2 (cross down)
        signal_diff = df.get_column(signal_name).diff()
        cross_signal_values = [
            1 if v == 2 else (-1 if v == -2 else v) for v in signal_diff.to_list()
        ]
        cross_signal = nw.new_series(signal_name, cross_signal_values, backend=native_ns)
        df = df.with_columns(cross_signal)

    return df.select(fast_col, slow_col, signal_name)


BMSB_FAST = MA(length=21 * 7, mode="ema")  # 21 weeks
BMSB_SLOW = MA(length=20 * 7, mode="sma")  # 20 weeks


@nw.narwhalify(eager_only=True)
def bull_market_support_band(series: nw.Series, *, only_crosses: bool = False) -> nw.DataFrame:
    """
    Bull Market Support Band.

    Reference:
        1. Benjamin Cowen - https://intothecryptoverse.com/
    """
    return nw.from_native(
        ma_cross(series, BMSB_FAST, BMSB_SLOW, signal_name="BMSB", only_crosses=only_crosses), eager_only=True
    )


bmsb = bull_market_support_band

LARSSON_FAST = MA(length=35, mode="ema")
LARSSON_SLOW = MA(length=50, mode="ema")


@nw.narwhalify(eager_only=True)
def larsson_line(series: nw.Series, *, only_crosses: bool = False) -> nw.DataFrame:
    """
    Larsson Line.

    Reference:
        1. CTO Larsen - https://www.ctolarsson.com/
    """
    return nw.from_native(
        ma_cross(series, LARSSON_FAST, LARSSON_SLOW, signal_name="LL", only_crosses=only_crosses), eager_only=True
    )
