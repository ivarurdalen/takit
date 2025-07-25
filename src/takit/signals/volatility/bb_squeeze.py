import pandas as pd
from pydantic import PositiveFloat, PositiveInt, validate_call

from takit.indicators.volatility import bollinger_bands
from takit.validation import config_dict


@validate_call(config=config_dict)
def bollinger_bands_squeeze(
    series: pd.Series,
    width_lookback_length: PositiveInt = 125,
    bb_length: PositiveInt = 20,
    n_std_dev: PositiveFloat = 2.0,
) -> pd.DataFrame:
    """
    Bollinger Bands Squeeze.

    Reference:
        1. John Bollinger. Bollinger on Bollinger Bands.

    Args:
        series: Input series
        width_lookback_length: Length of the window for the lowest width
        bb_length: Length of the window for the Bollinger Bands
        n_std_dev: Number of standard deviations from the moving average

    Returns:
        DataFrame with BBM, BBL, BBU, BBW, BBSQZ columns.
    """
    df = bollinger_bands(series, bb_length, n_std_dev, include_width=True)

    width_col_name = f"BBW{bb_length}_{n_std_dev}"
    lowest_width = df[width_col_name].rolling(window=width_lookback_length).min()

    df[f"BBSQZ{bb_length}_{n_std_dev}_{width_lookback_length}"] = (df[width_col_name] == lowest_width).astype(int)
    return df


bb_squeeze = bollinger_bands_squeeze
