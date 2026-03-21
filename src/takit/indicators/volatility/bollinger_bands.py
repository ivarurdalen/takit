from __future__ import annotations

import narwhals as nw
from pydantic import NonNegativeInt, PositiveFloat, PositiveInt, validate_call

from takit.indicators.trend.ma import simple_moving_average
from takit.validation import config_dict


@nw.narwhalify(eager_only=True)
@validate_call(config=config_dict)
def bollinger_bands(
    series: nw.Series,
    length: PositiveInt = 20,
    n_std_dev: PositiveFloat = 2.0,
    ddof: NonNegativeInt = 0,
    *,
    include_width: bool = False,
    include_percentage: bool = False,
) -> nw.DataFrame:
    """
    Bollinger Bands (BB).

    Width (BBW) and percentage (BBP) can be included as additional columns.

    Reference:
        1. https://www.tradingview.com/support/solutions/43000501840-bollinger-bands-bb/
        2. https://www.tradingview.com/support/solutions/43000501971-bollinger-bands-b-b/

    Equations:
        1. BBM = MA(series, length)
        2. BBL = BBM - n_std_dev * STD(series, length)
        3. BBU = BBM + n_std_dev * STD(series, length)
        4. BBW = (BBU - BBL) / BBM
        5. BBP = (series - BBL) / (BBU - BBL)

    Args:
        series: Input series
        length: Length of the window
        n_std_dev: Number of standard deviations from the moving average
        ddof: Delta degrees of freedom
        include_width: Include the width of the bands
        include_percentage: Include the percentage of the bands

    Returns:
        BBM, BBL, BBU columns. BBW and BBP columns if include_width and include_percentage are True.
    """
    basis = nw.from_native(simple_moving_average(series, length), series_only=True).alias(f"BBM{length}_{n_std_dev}")
    std = series.rolling_std(window_size=length, ddof=ddof).alias(f"STD{length}")
    lower_band = (basis - n_std_dev * std).alias(f"BBL{length}_{n_std_dev}")
    upper_band = (basis + n_std_dev * std).alias(f"BBU{length}_{n_std_dev}")

    cols = [basis, lower_band, upper_band]
    if include_width:
        cols.append(((upper_band - lower_band) / basis).alias(f"BBW{length}_{n_std_dev}"))
    if include_percentage:
        cols.append(((series - lower_band) / (upper_band - lower_band)).alias(f"BBP{length}_{n_std_dev}"))

    native_ns = nw.get_native_namespace(series)
    data = {s.name: s.to_native() for s in cols}
    return nw.from_native(native_ns.DataFrame(data), eager_only=True)


bb = bollinger_bands
