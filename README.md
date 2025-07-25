# takit

[![Ruff][ruff-badge]](https://github.com/astral-sh/ruff)
[![PyPI][pypi-badge]](https://pypi.org/project/takit/)
[![Python versions][python-versions-badge]](https://github.com/ivarurdalen/takit)
[![MIT License Badge][license-badge]][license]

Technical analysis library for Python.

## Definitions

The library is structured around the following definitions:

- **Indicator**: Produces a continuous value based on the input data to indicate trend, momentum, volatility, etc.
- **Signal**: Produces a discrete value like 1, 0, -1 to signal trend change, overbought, oversold, etc.

These functions are located in the different subpackages `indicators` and `signals`.

## Features

- Validation of input pandas Series / DataFrames to indicators and signals
- Ready to use signal calculations based on the indicator values

## Installation

```bash
pip install takit
# or with optional group `data` for fetching data
pip install "takit[data]"
```

## Usage

To use the following example you need to install the library with the optional group `data`.

```python
import pandas as pd

import takit

# Fetch data
ohlc = takit.data.fetch_data(
    data_source="binance", ticker="BTCUSDT", interval="1d", start="2023-01-01", end="2025-07-01"
)

# Calculate indicators
df = pd.concat(
    [
        ohlc,
        takit.bb(ohlc["close"], include_width=True, include_percentage=True),
        takit.wvf(ohlc),
        takit.rsi(ohlc["close"]),
    ],
    axis=1,
)

print(df.tail(40))
```

[ruff-badge]: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json
[license]: ./LICENSE
[license-badge]: https://img.shields.io/badge/License-MIT-blue.svg
[python-versions-badge]: https://img.shields.io/pypi/pyversions/takit.svg
[pypi-badge]: https://img.shields.io/pypi/v/takit
