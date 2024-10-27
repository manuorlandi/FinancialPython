import sys
from pathlib import Path

# Add the project root to the Python path

# Now import the required modules
import datetime as dt
import polars as pl
from __init__ import PROJECT_ROOT, DATA_PATH

# Rest of your imports and code...

print(f"PROJECT_ROOT: {PROJECT_ROOT}")
print(f"DATA_PATH: {DATA_PATH}")

# PARQUETS DEFINITION
TRAIN_FILENAME = "train.parquet"
TEST_FILENAME = "test.parquet"


# PARQUETS DEFINITION
TRAIN_FILENAME = "train.parquet"
TEST_FILENAME = "test.parquet"

# PARAMS
TEST_SIZE = 720


def split_train_test(
    df: pl.DataFrame,
    column_date: str = "date",
    *,
    threshold_date: dt.datetime = None,
    test_size: int = None
):
    """
    Split a DataFrame into train and test sets based on date or amount.

    Args:
        df (pl.DataFrame): The input DataFrame.
        column_date (str): The name of the date column. Defaults to "date".
        threshold_date (dt.datetime, optional): The date to split on. If provided, splits by date.
        test_size (int, optional): The number of samples per symbol for the test set. If provided, splits by amount.

    Returns:
        tuple: (train DataFrame, test DataFrame)

    Raises:
        ValueError: If neither threshold_date nor test_size is provided, or if both are provided.
    """
    df = df.sort(column_date)
    if (threshold_date is None) == (test_size is None):
        raise ValueError("Exactly one of threshold_date or test_size must be provided")

    if threshold_date is not None:
        return (
            df.filter(pl.col(column_date) < threshold_date),
            df.filter(pl.col(column_date) >= threshold_date)
        )

    total_rows = df.shape[0]
    n_symbols = df["symbol"].n_unique()
    test_rows = test_size * n_symbols
    
    return df.slice(0, total_rows - test_rows), df.slice(total_rows - test_rows)


if __name__ == "__main__":
    df = pl.read_parquet(DATA_PATH / "feature_engineering.parquet")

    df_train, df_test = split_train_test(
        df=df,
        column_date="date",
        test_size=TEST_SIZE,
    )

    df_train.write_parquet(DATA_PATH / TRAIN_FILENAME)
    df_test.write_parquet(DATA_PATH / TEST_FILENAME)