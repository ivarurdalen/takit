from typing import Annotated

import pandas as pd
from pydantic import PositiveInt, StringConstraints, validate_call

from takit.indicators.trend import ma
from takit.validation import config_dict


@validate_call(config=config_dict)
def moving_average_deviation(
    series: pd.Series, length: PositiveInt = 140, mode: Annotated[str, StringConstraints(pattern="sma|ema")] = "sma"
) -> pd.Series:
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
    return (series / getattr(ma, mode)(series, length=length)).rename(f"MAD_{mode.upper()}{length}")


mad = moving_average_deviation
