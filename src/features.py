"""Cleaning helpers and the time-based split."""

from __future__ import annotations

import pandas as pd


def time_split(
    frame: pd.DataFrame, holdout_start: str = "2025-09-01"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split on date: everything before ``holdout_start`` is train.

    The hidden validation set is Nov-Dec (the future), so model selection has
    to simulate predicting forward in time, not interpolating a random sample.
    """
    df = frame.copy()
    df["date"] = pd.to_datetime(df["date"])
    train = df[df["date"] < holdout_start].copy()
    holdout = df[df["date"] >= holdout_start].copy()
    return train, holdout
