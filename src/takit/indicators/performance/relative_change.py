import narwhals as nw
from pydantic import validate_call

from takit.validation import config_dict, validate_series


@nw.narwhalify(eager_only=True)
@validate_call(config=config_dict)
def relative_change(series: nw.Series, start: str | None = None) -> nw.Series:
    """
    Relative Change (RC).

    Calculate the relative change of a series from a start date.

    Args:
        series: Input series with DatetimeIndex
        start: Start date (ISO 8601 string). If None, use the first value.
            Note: Date filtering requires a pandas Series with a DatetimeIndex.
    """
    series = validate_series(series, min_length=2)

    if start is not None:
        # Date filtering via index is a pandas-specific operation.
        # Fall back to native for this operation.
        native = nw.to_native(series)
        native = native.loc[start:]
        series = nw.from_native(native, series_only=True)

    first_value = series[0]
    return (series / first_value - 1).alias("RC")
