from enum import Enum


class DataSource(str, Enum):
    BINANCE = "binance"


class Interval(str, Enum):
    D1 = "1d"
    H1 = "1h"
