import pandas as pd

from takit.indicators.trend import simple_moving_average
from takit.signals.trend.ma_cross import MA, ma_cross

PI_CYCLE_TOP_FAST = MA(length=111)
PI_CYCLE_TOP_SLOW = MA(length=350, multiplier=2)


def pi_cycle_top(series: pd.Series, *, only_crosses: bool = True) -> pd.DataFrame:
    """
    PI Cycle Top.

    Reference:
        https://www.lookintobitcoin.com/charts/pi-cycle-top-indicator/
    """
    return ma_cross(series, PI_CYCLE_TOP_FAST, PI_CYCLE_TOP_SLOW, only_crosses=only_crosses)


def mayer_multiple(series: pd.Series) -> pd.DataFrame:
    """
    Mayer Multiple.

    Reference:
        https://studio.glassnode.com/workbench/btc-mayer-multiple
    """
    df = series.to_frame()

    df["MMA"] = simple_moving_average(series, 200)
    df["MMAx0.8"] = df["MMA"] * 0.8
    df["MMAx2"] = df["MMA"] * 2
    df["MM"] = series / df["MMA"]

    return df.iloc[:, -4:]


DAYS_IN_A_YEAR = 365


def two_yr_ma(series: pd.Series) -> pd.DataFrame:
    """
    Two Year MA a.k.a Bitcoin Investor Tool.

    Reference:
        https://www.lookintobitcoin.com/charts/bitcoin-investor-tool/
    """
    df = series.to_frame()

    df["SMA2Y"] = simple_moving_average(series, 2 * DAYS_IN_A_YEAR, min_periods=1)
    df["SMA2Yx5"] = df["SMA2Y"] * 5

    return df.iloc[:, -2:]


def golden_ratio(series: pd.Series) -> pd.DataFrame:
    """
    Golden Ratio.

    Reference:
        https://charts.bitbo.io/golden-ratio/
    """
    df = series.to_frame()

    df["MA350"] = simple_moving_average(series, 350, min_periods=1)
    df["MA350x1.6"] = df["MA350"] * 1.6
    df["MA350x2"] = df["MA350"] * 2
    df["MA350x3"] = df["MA350"] * 3
    return df.iloc[:, -4:]
