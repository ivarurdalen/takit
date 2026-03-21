from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

from pydantic import StringConstraints, validate_call

from takit.enums import DataSource, Interval
from takit.validation import config_dict

DEFAULT_DATA_FOLDER = Path.cwd() / "data"


@validate_call(config=config_dict)
def fetch_data(
    data_source: DataSource,
    ticker: str,
    interval: Interval,
    start: Annotated[str, StringConstraints(pattern=r"^\d{4}-\d{2}-\d{2}$")],
    end: Annotated[str, StringConstraints(pattern=r"^\d{4}-\d{2}-\d{2}$")],
    *,
    data_folder: Path = DEFAULT_DATA_FOLDER,
    output_format: Literal["pandas", "polars"] = "pandas",
):
    """
    Fetch data according to the given parameters.

    Args:
        data_source: The data source to fetch data from
        ticker: The ticker to fetch data for
        interval: The interval (e.g. 1h, 1d)
        start: The ISO 8601 start date
        end: The ISO 8601 end date
        data_folder: The folder to store the data in
        output_format: Output DataFrame format ("pandas" or "polars")
    """
    import pandas as pd

    data_folder.mkdir(parents=True, exist_ok=True)
    filepath = data_folder / f"{data_source.value}_{ticker}_{interval.value}_{start}_{end}.parquet"
    if filepath.exists():
        df = pd.read_parquet(filepath)
    elif data_source == DataSource.BINANCE:
        df = _fetch_binance_data(ticker, interval, start, end)
        df.to_parquet(filepath)
    else:
        raise ValueError(f"Unknown data source: {data_source}")

    if output_format == "polars":
        import polars as pl

        return pl.from_pandas(df.reset_index())
    return df


def _fetch_binance_data(ticker: str, interval: Interval, start: str, end: str):
    """Fetch data from Binance for the given ticker and interval."""
    from .binance_client import BinanceClient

    client = BinanceClient()
    return client.get_df(ticker, interval=interval.value, start=start, end=end)
