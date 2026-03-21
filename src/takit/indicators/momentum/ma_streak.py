from __future__ import annotations

from typing import Annotated

import narwhals as nw
from pydantic import PositiveInt, StringConstraints, validate_call

from takit.indicators.trend import ma
from takit.util.base import value_streak
from takit.validation import config_dict


@nw.narwhalify(eager_only=True)
@validate_call(config=config_dict)
def moving_average_streak(
    series: nw.Series, length: PositiveInt = 20, mode: Annotated[str, StringConstraints(pattern="sma|ema")] = "sma"
) -> nw.DataFrame:
    """
    Moving average streak.

    Args:
        series: Input series
        length: Length of the moving average window
        mode: Mode of the moving average (sma or ema)

    Returns:
        DataFrame with moving average and streak values
    """
    col_name = f"{mode.upper()}{length}"
    ma_values = nw.from_native(getattr(ma, mode)(series, length=length), series_only=True).alias(col_name)

    # Compute direction of change: 1 for increase/unchanged, -1 for decrease
    change = ma_values.diff()
    temp_df = change.to_frame()
    direction = temp_df.select(
        nw.when(nw.col(change.name) >= 0).then(1).otherwise(-1).alias("direction")
    ).get_column("direction")

    streak = value_streak(direction, negative_streaks=True).alias(f"{col_name}STREAK")

    native_ns = nw.get_native_namespace(series)
    return nw.from_native(
        native_ns.DataFrame({col_name: ma_values.to_native(), streak.name: streak.to_native()}), eager_only=True
    )


ma_streak = moving_average_streak
