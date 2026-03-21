from datetime import datetime, timezone


def to_milliseconds(date: str) -> int:
    """Convert an ISO 8601 date string to milliseconds since the epoch."""
    dt = datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)
