import pandas as pd
from binance.spot import Spot

from takit.util.date import to_milliseconds

DEFAULT_LIMIT = 1000


class BinanceClient:
    def __init__(self):
        self.client = Spot()

    def get_df(self, symbol: str, interval: str, start: str, end: str) -> pd.DataFrame:
        """
        Get a DataFrame of the klines for the given symbol and interval.

        Args:
            symbol: The symbol to get the klines for
            interval: The interval of the klines
            start: The start time of the klines
            end: The end time of the klines
        """
        symbol = symbol.upper().replace("/", "")
        start_time, end_time = to_milliseconds(start), to_milliseconds(end)
        data = []
        while start_time <= end_time:
            response = self.client.klines(
                symbol=symbol, interval=interval, limit=DEFAULT_LIMIT, startTime=start_time, endTime=end_time
            )
            data.extend(response)
            if len(response) < DEFAULT_LIMIT:
                break
            start_time = data[-1][0] + 1

        return self._wrangle_data(symbol, data)

    def _wrangle_data(self, symbol: str, data: list[list]) -> pd.DataFrame:
        data = [candle[:6] for candle in data]  # Only keep the first 6 columns
        data = [[pd.Timestamp(candle[0], unit="ms"), *candle[1:]] for candle in data]
        df = pd.DataFrame(data, columns=["date", "open", "high", "low", "close", "volume"])
        df = df.set_index("date").sort_index()
        df.index = pd.to_datetime(df.index)
        return df.astype(float)
