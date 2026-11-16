"""Shared data loading and feature preparation for the FM Energy Workshop.

The feature logic lives here so the step scripts (Step 2 onward) can reuse it
without duplicating code.
"""
from pathlib import Path
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = REPO_ROOT / "Data" / "one_meter.csv"

FEATURES = ["hour_of_day", "day_of_week", "lag_1", "lag_2", "lag_3", "lag_24", "lag_168"]
TARGET = "electricity_kwh"
DROP_FIRST = 168  # rows that lack full history for the 168-hour lag


def load_series():
    """Load the raw CSV as a clean, time-sorted hourly Series of electricity_kwh."""
    df = pd.read_csv(CSV_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    df = df.set_index("timestamp")
    return df[TARGET]


def build_features(ser):
    """Build the feature table for predicting electricity_kwh at hour t.

    All features are available at prediction time (no leakage):
      hour_of_day  - 0..23 from the timestamp
      day_of_week  - Monday=0 .. Sunday=6 from the timestamp
      lag_1, lag_2, lag_3, lag_24, lag_168 - readings 1,2,3,24,168 hours earlier
    Then drops the first 168 rows that lack full history for the 168-hour lag.
    """
    data = pd.DataFrame(index=ser.index)
    data["hour_of_day"] = ser.index.hour
    data["day_of_week"] = ser.index.dayofweek  # Monday=0 .. Sunday=6
    data["lag_1"] = ser.shift(1)
    data["lag_2"] = ser.shift(2)
    data["lag_3"] = ser.shift(3)
    data["lag_24"] = ser.shift(24)
    data["lag_168"] = ser.shift(168)
    data[TARGET] = ser
    # Drop the first 168 rows that lack full history for the 168-hour lag.
    data = data.iloc[DROP_FIRST:].copy()
    return data


def load_prepared():
    """Return the prepared feature+target table, ready for modeling."""
    return build_features(load_series())


TEST_HOURS = 336  # last two weeks held back as the test set


def split_train_test(data, test_hours=TEST_HOURS):
    """Time-based split: the last `test_hours` rows are the test set the model
    never sees during training; everything before is the training set.
    Splitting by position keeps the temporal order (no random shuffling).
    """
    test = data.iloc[-test_hours:]
    train = data.iloc[:-test_hours]
    return train, test


def make_xy(df):
    """Split a prepared table into feature matrix X and target vector y."""
    return df[FEATURES], df[TARGET]
