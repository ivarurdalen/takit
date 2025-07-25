from typing import Annotated

import pandera.pandas as pa
from pydantic import ConfigDict, StringConstraints

PriceCol = Annotated[str, StringConstraints(pattern="close|high|low|open")]

ordered_index_schema = pa.Index(
    checks=[
        pa.Check(
            lambda i: i.is_monotonic_increasing, element_wise=False, error="Index must be monotonically increasing"
        )
    ],
    nullable=False,
    unique=True,
)

ohlc_column_schema = pa.Column(float, checks=[pa.Check.greater_than(0)])


def ohlc_schema(min_length: int = 1) -> pa.DataFrameSchema:
    """
    Schema for OHLC data.

    Args:
        min_length: Minimum length of the series

    Returns:
        Schema for OHLC data
    """
    return pa.DataFrameSchema(
        columns={
            "open": ohlc_column_schema,
            "high": ohlc_column_schema,
            "low": ohlc_column_schema,
            "close": ohlc_column_schema,
        },
        checks=[
            pa.Check(
                lambda df: df.shape[0] > min_length,
                element_wise=False,
                error=f"DataFrame must have at least {min_length} rows.",
            )
        ],
        index=ordered_index_schema,
    )


def ta_series_schema(min_length: int = 1, dtype: type = float) -> pa.SeriesSchema:
    """
    Return a schema for a series to be used in technical analysis.

    Args:
        min_length: Minimum length of the series
        dtype: Data type of the series
    """
    return pa.SeriesSchema(
        dtype,
        checks=[
            pa.Check.greater_than_or_equal_to(0),
            pa.Check(
                lambda s: s.size > min_length,
                element_wise=False,
                error=f"Series must have at least {min_length} values.",
            ),
        ],
        index=ordered_index_schema,
        nullable=False,
    )


config_dict = ConfigDict(arbitrary_types_allowed=True)
