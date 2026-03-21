from __future__ import annotations

from typing import Annotated

import narwhals as nw
from pydantic import PositiveInt, StringConstraints, validate_call

from takit.indicators.trend import ma
from takit.validation import config_dict


@nw.narwhalify(eager_only=True)
@validate_call(config=config_dict)
def moving_average_deviation(
    series: nw.Series, length: PositiveInt = 140, mode: Annotated[str, StringConstraints(pattern="sma|ema")] = "sma"
) -> nw.Series:
    """
    Moving average deviation (MAD).

    Other names: Bias / Disparity Index.

    Equations:
       1. MAD = (series / MA(series, length))

    Args:
        series: Input series
        length: Length of the moving average window
        mode: Mode of the moving average (sma or ema)

    Returns:
        MAD values of the input series
    """
    return (series / getattr(ma, mode)(series, length=length)).alias(f"MAD_{mode.upper()}{length}")


mad = moving_average_deviation
