from pathlib import Path
from typing import Annotated

import pandas as pd
from pydantic import StringConstraints, validate_call

from takit.enums import DataSource, Interval
from takit.validation import config_dict

from .binance_client import BinanceClient

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
) -> pd.DataFrame:
    """
    Fetch data according to the given parameters.

    Args:
        data_source: The data source to fetch data from
        ticker: The ticker to fetch data for
        interval: The interval (e.g. 1h, 1d)
        start: The ISO 8601 start date
        end: The ISO 8601 end date
        data_folder: The folder to store the data in
    """
    data_folder.mkdir(parents=True, exist_ok=True)
    filepath = data_folder / f"{data_source.value}_{ticker}_{interval.value}_{start}_{end}.parquet"
    if filepath.exists():
        return pd.read_parquet(filepath)
    if data_source == DataSource.BINANCE:
        df = fetch_binance_data(ticker, interval, start, end)
    else:
        raise ValueError(f"Unknown data source: {data_source}")
    df.to_parquet(filepath)
    return df


def fetch_binance_data(ticker: str, interval: Interval, start: str, end: str) -> pd.DataFrame:
    """Fetch data from Binance for the given ticker and interval."""
    client = BinanceClient()
    return client.get_df(ticker, interval=interval.value, start=start, end=end)
