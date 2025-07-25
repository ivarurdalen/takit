from pathlib import Path
from typing import Annotated

import pandas as pd
import typer

import takit
from takit.enums import DataSource, Interval
from takit.signals.trend.ma_cross import MA

app = typer.Typer()

DEFAULT_DATA_SOURCE = DataSource.BINANCE
DEFAULT_TICKER = "BTCUSDT"
DEFAULT_INTERVAL = Interval.D1
DEFAULT_START = (pd.Timestamp.utcnow() - pd.Timedelta(days=30)).date().isoformat()
DEFAULT_END = (pd.Timestamp.utcnow() - pd.Timedelta(days=1)).date().isoformat()

DATA_FOLDER = Path(__file__).parent / "data"


@app.command()
def ta(
    indicators: Annotated[list[str], typer.Argument()],
    data_source: Annotated[DataSource, typer.Option()] = DEFAULT_DATA_SOURCE,
    ticker: Annotated[str, typer.Option()] = DEFAULT_TICKER,
    interval: Annotated[Interval, typer.Option()] = DEFAULT_INTERVAL,
    start: Annotated[str, typer.Option()] = DEFAULT_START,
    end: Annotated[str, typer.Option()] = DEFAULT_END,
    tail: Annotated[int, typer.Option()] = 30,
    only_trigger_rows: Annotated[bool, typer.Option(help="Only keep rows where the signal changed.")] = False,
    length: Annotated[int | None, typer.Option()] = None,
    compare: Annotated[bool, typer.Option(help="Compare with other libraries.")] = False,
):
    """Perform technical analysis on the given data and possibly compare it with other libraries."""
    print(f"Fetching data from {data_source} for {ticker} from {start} to {end}. Interval: {interval}")
    df = takit.data.fetch_data(data_source, ticker, interval, start, end)

    dfs = [df]
    for indicator in indicators:
        match indicator:
            case "rsi" | "relative_strength_index":
                length = 14 if length is None else length
                dfs.append(takit.rsi(df["close"], length).to_frame())
                if compare:
                    from bamboo_ta import relative_strength_index

                    dfs.append(relative_strength_index(df, period=length).rename(columns={"rsi": f"bta_RSI{length}"}))
                    from talib import RSI

                    dfs.append(
                        pd.DataFrame(
                            [{"talib_RSI{length}": RSI(df["close"].to_numpy(), timeperiod=length)}], index=df.index
                        )
                    )

            case "sma" | "simple_moving_average":
                length = 20 if length is None else length
                dfs.append(takit.sma(df["close"], length).to_frame())
                if compare:
                    from bamboo_ta import simple_moving_average

                    dfs.append(simple_moving_average(df, period=length).rename(columns={"sma": f"bta_SMA{length}"}))
                    from talib import SMA

                    dfs.append(
                        pd.DataFrame(
                            [{"talib_SMA{length}": SMA(df["close"].to_numpy(), timeperiod=length)}], index=df.index
                        )
                    )

            case "ema" | "exponential_moving_average":
                length = 20 if length is None else length
                dfs.append(takit.ema(df["close"], length).to_frame())
                if compare:
                    from bamboo_ta import exponential_moving_average

                    dfs.append(
                        exponential_moving_average(df, period=length).rename(columns={"ema": f"bta_EMA{length}"})
                    )
                    from talib import EMA

                    dfs.append(
                        pd.DataFrame(
                            [{"talib_EMA{length}": EMA(df["close"].to_numpy(), timeperiod=length)}], index=df.index
                        )
                    )

            case "mad" | "moving_average_deviation":
                length = 140 if length is None else length
                dfs.append(takit.mad(df["close"], length, "sma").to_frame())
                if compare:
                    from bamboo_ta import bias

                    dfs.append(bias(df, length=length).rename(columns={"bias": f"bta_MAD{length}"}))

            case "ma_streak" | "moving_average_streak":
                length = 20 if length is None else length
                dfs.append(takit.ma_streak(df["close"], length))

            case "atr":
                length = 14 if length is None else length
                dfs.append(takit.atr(df, length).to_frame())
                if compare:
                    from bamboo_ta import average_true_range
                    from talib import ATR

                    dfs.append(average_true_range(df, period=length).rename(columns={"atr": f"bta_ATR{length}"}))
                    dfs.append(
                        pd.DataFrame(
                            [
                                {
                                    "talib_ATR{length}": ATR(
                                        df["high"].to_numpy(),
                                        df["low"].to_numpy(),
                                        df["close"].to_numpy(),
                                        timeperiod=length,
                                    )
                                }
                            ],
                            index=df.index,
                        )
                    )

            case "bb" | "bollinger_bands":
                dfs.append(takit.bb(df["close"], include_width=True, include_percentage=True))
                if compare:
                    from bamboo_ta import bollinger_bands

                    df = bollinger_bands(df)
                    df.columns = [f"bta_BBL{length}", f"bta_BBM{length}", f"bta_BBU{length}"]
                    dfs.append(df)

            case "ma_cross":
                fast_ma = MA(length=20)
                slow_ma = MA(length=50)
                dfs.append(takit.ma_cross(df["close"], fast_ma, slow_ma))

            case "pi_cycle_top" | "bull_market_support_band" | "bmsb" | "larsson_line":
                df = getattr(takit, indicator)(df["close"], only_crosses=only_trigger_rows)
                df = df.dropna(axis=0, how="any")
                if only_trigger_rows:
                    df = df[df.iloc[:, -1] != 0]
                dfs.append(df)

            case "wvf" | "williams_vix_fix":
                dfs.append(takit.wvf(df))

            case _:
                dfs.append(getattr(takit, indicator)(df["close"]))

    output = pd.concat(dfs, axis=1).reset_index()
    if tail:
        output = output.tail(tail)
    print(output)


if __name__ == "__main__":
    app()
