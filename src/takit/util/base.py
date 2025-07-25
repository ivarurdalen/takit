import pandas as pd


def value_streak(series: pd.Series, *, negative_streaks: bool = False) -> pd.Series:
    """
    Value streak.

    Calculates the cumulative count of consecutive equal values.
    For example, [1, 1, 2, 2, 2, 3] -> [1, 2, 1, 2, 3, 1]

    Args:
        series: Input series
        negative_streaks: Whether make streaks of negative values negative

    Returns:
        Streak values of the input series
    """
    # Identify where value changes
    groups = (series != series.shift()).cumsum()
    # Count the streak within each group
    streak = series.groupby(groups).cumcount() + 1

    if negative_streaks:
        # For negative values, make streak count negative
        return streak.where(series >= 0, -streak)
    return streak
