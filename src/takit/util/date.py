import pandas as pd


def to_milliseconds(date: str) -> int:
    """Convert an ISO 8601 date string to milliseconds since the epoch."""
    return int(pd.Timestamp(date).timestamp() * 1000)
