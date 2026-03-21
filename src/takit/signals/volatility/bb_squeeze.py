import narwhals as nw
from pydantic import PositiveFloat, PositiveInt, validate_call

from takit.indicators.volatility import bollinger_bands
from takit.util.rolling import rolling_min
from takit.validation import config_dict


@nw.narwhalify(eager_only=True)
@validate_call(config=config_dict)
def bollinger_bands_squeeze(
    series: nw.Series,
    width_lookback_length: PositiveInt = 125,
    bb_length: PositiveInt = 20,
    n_std_dev: PositiveFloat = 2.0,
) -> nw.DataFrame:
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
    df = nw.from_native(bollinger_bands(series, bb_length, n_std_dev, include_width=True), eager_only=True)

    width_col_name = f"BBW{bb_length}_{n_std_dev}"
    sqz_col_name = f"BBSQZ{bb_length}_{n_std_dev}_{width_lookback_length}"

    width = df.get_column(width_col_name)
    lowest_width = rolling_min(width, width_lookback_length)

    squeeze = nw.new_series(
        sqz_col_name,
        [1 if w == lw else 0 for w, lw in zip(width.to_list(), lowest_width.to_list(), strict=True)],
        backend=nw.get_native_namespace(series),
    )

    native_ns = nw.get_native_namespace(series)
    data = {col: df.get_column(col).to_native() for col in df.columns}
    data[sqz_col_name] = squeeze.to_native()
    return nw.from_native(native_ns.DataFrame(data), eager_only=True)


bb_squeeze = bollinger_bands_squeeze
