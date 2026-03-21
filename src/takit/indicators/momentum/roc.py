import narwhals as nw
from pydantic import PositiveInt, validate_call

from takit.validation import config_dict, validate_series


@nw.narwhalify(eager_only=True)
@validate_call(config=config_dict)
def rate_of_change(series: nw.Series, length: PositiveInt = 14) -> nw.Series:
    """
    Rate of Change (ROC).

    Reference:
        1. https://www.tradingview.com/support/solutions/43000502345-rate-of-change-roc/

    Equations:
        1. ROC = 100 * (series - series[n]) / series[n]

    Args:
        series: Input series
        length: Lookback period

    Returns:
        ROC values
    """
    series = validate_series(series, min_length=length)
    return (100 * (series - series.shift(length)) / series.shift(length)).alias(f"ROC{length}")


roc = rate_of_change
