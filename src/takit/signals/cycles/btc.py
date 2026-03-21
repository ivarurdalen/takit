import narwhals as nw

from takit.indicators.trend import simple_moving_average
from takit.signals.trend.ma_cross import MA, ma_cross

PI_CYCLE_TOP_FAST = MA(length=111)
PI_CYCLE_TOP_SLOW = MA(length=350, multiplier=2)


@nw.narwhalify(eager_only=True)
def pi_cycle_top(series: nw.Series, *, only_crosses: bool = True) -> nw.DataFrame:
    """
    PI Cycle Top.

    Reference:
        https://www.lookintobitcoin.com/charts/pi-cycle-top-indicator/
    """
    return nw.from_native(
        ma_cross(series, PI_CYCLE_TOP_FAST, PI_CYCLE_TOP_SLOW, only_crosses=only_crosses), eager_only=True
    )


@nw.narwhalify(eager_only=True)
def mayer_multiple(series: nw.Series) -> nw.DataFrame:
    """
    Mayer Multiple.

    Reference:
        https://studio.glassnode.com/workbench/btc-mayer-multiple
    """
    sma200 = nw.from_native(simple_moving_average(series, 200), series_only=True).alias("MMA")
    sma200_x08 = (sma200 * 0.8).alias("MMAx0.8")
    sma200_x2 = (sma200 * 2).alias("MMAx2")
    mm = (series / sma200).alias("MM")

    native_ns = nw.get_native_namespace(series)
    data = {s.name: s.to_native() for s in [sma200, sma200_x08, sma200_x2, mm]}
    return nw.from_native(native_ns.DataFrame(data), eager_only=True)


DAYS_IN_A_YEAR = 365


@nw.narwhalify(eager_only=True)
def two_yr_ma(series: nw.Series) -> nw.DataFrame:
    """
    Two Year MA a.k.a Bitcoin Investor Tool.

    Reference:
        https://www.lookintobitcoin.com/charts/bitcoin-investor-tool/
    """
    sma2y = nw.from_native(
        simple_moving_average(series, 2 * DAYS_IN_A_YEAR, min_periods=1), series_only=True
    ).alias("SMA2Y")
    sma2y_x5 = (sma2y * 5).alias("SMA2Yx5")

    native_ns = nw.get_native_namespace(series)
    data = {s.name: s.to_native() for s in [sma2y, sma2y_x5]}
    return nw.from_native(native_ns.DataFrame(data), eager_only=True)


@nw.narwhalify(eager_only=True)
def golden_ratio(series: nw.Series) -> nw.DataFrame:
    """
    Golden Ratio.

    Reference:
        https://charts.bitbo.io/golden-ratio/
    """
    ma350 = nw.from_native(
        simple_moving_average(series, 350, min_periods=1), series_only=True
    ).alias("MA350")
    ma350_x16 = (ma350 * 1.6).alias("MA350x1.6")
    ma350_x2 = (ma350 * 2).alias("MA350x2")
    ma350_x3 = (ma350 * 3).alias("MA350x3")

    native_ns = nw.get_native_namespace(series)
    data = {s.name: s.to_native() for s in [ma350, ma350_x16, ma350_x2, ma350_x3]}
    return nw.from_native(native_ns.DataFrame(data), eager_only=True)
