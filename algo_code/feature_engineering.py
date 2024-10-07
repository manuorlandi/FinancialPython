import sys
import os
import numpy as np
import datetime as dt
import polars as pl
import yfinance as yf

from sklearn.model_selection import TimeSeriesSplit, train_test_split
from pathlib import Path
from pybit.unified_trading import HTTP

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fin_utilities
import matplotlib.pyplot as plt
import plotly.express as px
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.stattools import pacf, acf
from scipy.stats import norm
from typing import Tuple, List
from fredapi import Fred

fred = Fred(api_key="a2bdf7d81b8f0362f82e976bb92fbdb4")

cfg = fin_utilities.__cfg_reading("pred")


PROJECT_DIR = eval(cfg["PROJECT_PATH"])
DATA_PATH = PROJECT_DIR / cfg["DATA_FOLDER"]

START_DATE = dt.datetime(2022, 12, 28)
print(START_DATE)
END_DATE = dt.datetime.now()
print(END_DATE)


def split_target_features(
    df: pl.DataFrame,
    feat_to_exclude: list[str] = [],
    target_var: str = "signal",
) -> Tuple[pl.DataFrame, pl.Series]:
    """
    Given a DataFrame, split df into features and target.

    Args:
        df: dataframe to be split
        feat_to_exclude: list of columns to exclude from features
        target_var: variable target

    Returns:
        Tuple[pl.DataFrame, pl.Series]: feature and target data
    """
    # Select columns that are not in feat_to_exclude and not the target_var
    features = df.select(
        [
            col
            for col in df.columns
            if col not in (feat_to_exclude + [target_var])
        ]
    )
    labels = df.select([target_var])

    return features, labels


def pct_change_scale(col: str, suffix: str) -> pl.Expr:
    return pl.col(col).pct_change().alias(f"{col}{suffix}")


def z_score_scale(col: str, suffix: str) -> pl.Expr:
    return ((pl.col(col) - pl.col(col).mean()) / pl.col(col).std()).alias(
        f"{col}{suffix}"
    )


def diff_scale(col: str, suffix: str) -> pl.Expr:
    return pl.col(col).diff().alias(f"{col}{suffix}")


def scale_features(
    df: pl.DataFrame,
    cols: str | list[str],
    type_scale: str = "pct_change",
    suffix: str = "_pct_change",
) -> pl.DataFrame:
    type_scale_accepted = {"pct_change", "z_score", "diff"}
    if type_scale not in type_scale_accepted:
        raise ValueError(f"type_scale must be one of {type_scale_accepted}")

    if isinstance(cols, str):
        cols = [cols]

    scale_functions = {
        "pct_change": pct_change_scale,
        "z_score": z_score_scale,
        "diff": diff_scale,
    }

    new_columns = [scale_functions[type_scale](col, suffix) for col in cols]

    return df.with_columns(new_columns)


def plot_histogram(
    df: pl.DataFrame,
    symbol: str,
    col: str,
    bins: int = 100,
):
    series_to_plot = df.filter(pl.col("symbol") == symbol)[col]

    # Create the histogram
    plt.figure(figsize=(10, 6))
    plt.hist(series_to_plot, bins=bins, edgecolor="black")
    plt.title(f"Histogram of {symbol.upper()}, column {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3)

    # Show the plot
    plt.show()


def calculate_rsi(
    df: pl.DataFrame,
    column: str = "close",
    window: int = 14,
) -> pl.DataFrame:
    # Calculate price changes
    delta = df[column].diff()

    # Separate gains and losses
    gains = delta.clip(lower_bound=0)
    losses = -delta.clip(upper_bound=0)

    # Calculate EMAs for gains and losses
    alpha = 1 / window
    gains_ema = gains.ewm_mean(alpha=alpha)
    losses_ema = losses.ewm_mean(alpha=alpha)

    # Calculate RS and RSI
    rs = gains_ema / losses_ema
    rsi = 100 - (100 / (1 + rs)).round(2)

    # Add RSI to the DataFrame
    return df.with_columns(rsi.alias(f"RSI_{window}"))


def calculate_rsi_by_symbol(
    df: pl.DataFrame,
    column: str = "close",
    window: int = 14,
) -> pl.DataFrame:
    symbol_list = df.partition_by("symbol", as_dict=False)
    return pl.concat(
        [calculate_rsi(symbol, column, window) for symbol in symbol_list]
    )


def calculate_macd(
    df: pl.DataFrame,
    column: str = "close",
    short_window: int = 12,
    long_window: int = 26,
    signal_window: int = 9,
) -> pl.DataFrame:
    # Calculate the short and long EMAs
    df = df.with_columns(
        [
            pl.col(column).ewm_mean(span=short_window).alias("ema_short"),
            pl.col(column).ewm_mean(span=long_window).alias("ema_long"),
        ]
    )

    # Calculate MACD
    df = df.with_columns(
        [(pl.col("ema_short") - pl.col("ema_long")).alias("MACD")]
    )

    # Calculate the signal line
    df = df.with_columns(
        [pl.col("MACD").ewm_mean(span=signal_window).alias("signal_line")]
    )

    # Calculate the MACD histogram (MACD - Signal)
    df = df.with_columns(
        [
            (pl.col("MACD") - pl.col("signal_line"))
            .round(2)
            .alias("macd_histogram")
        ]
    )

    return df


def calculate_macd_by_symbol(
    df: pl.DataFrame,
    column: str = "close",
    short_window: int = 12,
    long_window: int = 26,
    signal_window: int = 9,
) -> pl.DataFrame:
    symbol_list = df.partition_by("symbol", as_dict=False)
    return pl.concat(
        [
            calculate_macd(
                df=symbol,
                column=column,
                short_window=short_window,
                long_window=long_window,
                signal_window=signal_window,
            )
            for symbol in symbol_list
        ]
    )


def calculate_bollinger_bands(
    df: pl.DataFrame, column: str = "close", window: int = 20, k: float = 1.0
) -> pl.DataFrame:
    # Calculate rolling mean (SMA) and rolling standard deviation
    df = df.with_columns(
        [
            pl.col(column).rolling_mean(window).alias("sma"),
            pl.col(column).rolling_std(window, ddof=0).alias("std_dev"),
        ]
    )

    # Calculate upper and lower bands
    df = df.with_columns(
        [
            (pl.col("sma") + (k * pl.col("std_dev")))
            .alias("upper_band")
            .round(2),
            (pl.col("sma") - (k * pl.col("std_dev")))
            .alias("lower_band")
            .round(2),
        ]
    )

    return df


def calculate_bollinger_bands_by_symbol(
    df: pl.DataFrame, column: str = "close", window: int = 20, k: float = 1.0
) -> pl.DataFrame:
    symbol_list = df.partition_by("symbol", as_dict=False)
    return pl.concat(
        [
            calculate_bollinger_bands(
                df=symbol, column=column, window=window, k=k
            )
            for symbol in symbol_list
        ]
    )


def calculate_pacf(df, col, lags, alpha=0.05, keep_zero_lag=False, plot=False):
    z_score = norm.ppf((1 - alpha / 2)).round(2)
    pacf_values = pacf(x=df.select(col), nlags=lags)
    threshold = z_score / np.sqrt(len(df.select(col)))
    significant_lags = (np.where(np.abs(pacf_values) > threshold)[0]).tolist()
    if not keep_zero_lag:
        significant_lags.remove(0)

    if plot:
        fig, ax = plt.subplots(figsize=(15, 5))
        plot_pacf(
            df.select(col), lags=lags, zero=False, ax=ax
        )  # Specify number of lags to visualize
        plt.show()

    return significant_lags


def calculate_acf(df, col, lags, alpha=0.05, keep_zero_lag=False, plot=False):
    z_score = norm.ppf((1 - alpha / 2)).round(2)
    acf_values = acf(x=df.select(col), nlags=lags)
    threshold = z_score / np.sqrt(len(df.select(col)))
    significant_lags = (np.where(np.abs(acf_values) > threshold)[0]).tolist()
    if not keep_zero_lag:
        significant_lags.remove(0)

    if plot:
        fig, ax = plt.subplots(figsize=(15, 5))
        plt.stem(range(len(acf_values)), acf_values)
        plt.xlabel("Lags")
        plt.ylabel("ACF Value")
        plt.show()

    return significant_lags


def calculate_all_pacfs(df, col, lags, alpha, plot=False):
    lags_pacf_by_symbol = {}
    for symbol in df["symbol"].unique():
        tmp_df = df.filter(pl.col("symbol") == symbol)
        lags_pacf_by_symbol[symbol] = calculate_pacf(
            df=tmp_df, col=col, lags=lags, alpha=alpha
        )
    return lags_pacf_by_symbol


def calculate_all_acfs(df, col, lags, alpha, plot=False):
    lags_pacf_by_symbol = {}
    for symbol in df["symbol"].unique():
        tmp_df = df.filter(pl.col("symbol") == symbol)
        lags_pacf_by_symbol[symbol] = calculate_acf(
            df=tmp_df, col=col, lags=lags, alpha=alpha
        )
    return lags_pacf_by_symbol


def common_lags_by_symbol(lags_dict):
    common_lags = set(lags_dict.popitem()[1])

    for symbol in lags_dict.keys():
        common_lags = common_lags.intersection(set(lags_dict[symbol]))

    return list(common_lags)


def extract_date_features(df, date_col):
    df = df.with_columns(
        pl.col(date_col).dt.weekday().alias("dow"),
        pl.col(date_col).dt.month().alias("month"),
        pl.col(date_col).dt.hour().alias("hour"),
    )

    return df


def create_lagged_features(df, feature_col, significant_lags):
    for lag in significant_lags:
        df = df.with_columns(
            pl.col(feature_col).shift(lag).alias(f"{feature_col}_lag_{lag}")
        )
    return df


def create_lagged_features_by_symbol(df, feature_col, significant_lags):
    symbol_list = df.partition_by("symbol", as_dict=False)
    return pl.concat(
        [
            create_lagged_features(
                df=symbol,
                feature_col=feature_col,
                significant_lags=significant_lags,
            )
            for symbol in symbol_list
        ]
    )


def calculate_rolling_avgs(df: pl.DataFrame, moving_averages_list, on_columns):
    for window in moving_averages_list:
        for col in on_columns:
            df = df.with_columns(
                (pl.col(col).rolling_mean(window_size=window)).alias(
                    f"SMA_{col}_{window}"
                ),
            )
    return df


def calculate_rolling_avgs_by_symbol(
    df: pl.DataFrame,
    moving_averages_list,
    on_columns,
) -> pl.DataFrame:
    symbol_list = df.partition_by("symbol", as_dict=False)
    return pl.concat(
        [
            calculate_rolling_avgs(
                df=symbol,
                moving_averages_list=moving_averages_list,
                on_columns=on_columns,
            )
            for symbol in symbol_list
        ]
    )


# STOCK / VIX DATA
def generate_hourly_dates(start: str, end: str) -> pl.Series:
    # Generate date range
    date_range = pl.date_range(start=start, end=end, interval="1h")

    # Convert to Series
    return pl.Series("DATE", date_range)


def get_stock_data(tickers, start_date, end_date, interval="1h"):
    data = yf.download(
        tickers, start=start_date, end=end_date, interval=interval
    )
    pl_data = pl.from_pandas(data["Adj Close"].reset_index())
    pl_data = pl_data.rename({"Datetime": "date"})
    return pl_data


def get_vix_data(start_date, end_date):
    # Get stock market indices
    start = start_date.strftime("%Y-%m-%d")
    end = end_date.strftime("%Y-%m-%d")
    stock_tickers = ["^VIX"]
    stock_data = get_stock_data(stock_tickers, start, end)
    stock_data = stock_data.rename({"Adj Close": "VIX"})
    return stock_data


# DATA FROM FRED
def get_all_data(start_date, end_date):
    # Get stock market indices
    start = start_date.strftime("%Y-%m-%d")
    end = end_date.strftime("%Y-%m-%d")
    stock_tickers = ["^GSPC", "^IXIC"]
    stock_data = get_stock_data(stock_tickers, start, end)
    stock_data = stock_data.rename({"^GSPC": "SP500", "^IXIC": "NASDAQ"})
    return stock_data


def get_fred_data(series_ids, start_date, end_date):
    data = {}
    all_dates = set()
    for series_id in series_ids:
        series = fred.get_series(series_id, start_date, end_date)
        series = series.reset_index()
        series.columns = ["date", series_id]
        data[series_id] = series
        all_dates.update(series["date"])

    all_dates = sorted(all_dates)

    df = pl.DataFrame({"date": all_dates})
    df = df.with_columns(pl.col("date").dt.cast_time_unit("us").alias("date"))

    for series_id, series in data.items():
        tmp = pl.DataFrame(series)
        tmp = tmp.with_columns(
            pl.col("date").dt.cast_time_unit("us").alias("date")
        )
        df = df.join(tmp, on="date", how="left")

    # Fill missing values (you can choose a different strategy if needed)
    df = df.fill_null(strategy="forward")

    return df


def get_macroeconomic_data(start_date, end_date):
    series_ids = {
        "FEDFUNDS": "US_FED_RATE",
        "CPIAUCSL": "US_INFLATION",
        "GDP": "US_GDP",
        "UNRATE": "US_UNEMPLOYMENT",
    }
    data = pl.DataFrame(get_fred_data(series_ids.keys(), start_date, end_date))
    for old_name, new_name in series_ids.items():
        data = data.rename({old_name: new_name})
    return data


if __name__ == "__main__":
    df_orig = pl.read_parquet(DATA_PATH / "bybit_data.parquet")
    df = df_orig
    # a seconda della scala che voglio utilizzare, cambio il suffisso
    TYPE_SCALE_SUFFIXES = {
        "pct_change": "_pct_change",
        "z_score": "_z_score",
        "diff": "_diff",
    }
    type_scale = "pct_change"

    type_scale_suffix = TYPE_SCALE_SUFFIXES[type_scale]
    tgt_column = f"close{type_scale_suffix}"

    processed_dfs = []

    for symbol in df["symbol"].unique():
        # Filter the dataframe for the current symbol
        tmp_df = df.filter(pl.col("symbol") == symbol)
        # Apply scale_features to the filtered dataframe
        tmp_df = scale_features(
            tmp_df,
            ["close", "volume", "turnover"],
            type_scale=type_scale,
            suffix=type_scale_suffix,
        )
        # Append the processed dataframe to the list
        processed_dfs.append(tmp_df)

    # Concatenate all processed dataframes
    df = pl.concat(processed_dfs)

    # Drop any null values that might have been introduced
    df = df.drop_nulls()

    print("Calculate PACF lags...")
    lags_pacf = calculate_all_pacfs(
        df, f"close{type_scale_suffix}", 200, alpha=0.4
    )
    print("Calculate ACF lags...")
    lags_acf = calculate_all_acfs(
        df, f"close{type_scale_suffix}", 200, alpha=0.4
    )
    print("Calculate common PACF-ACF lags...")
    common_lags = list(
        set(common_lags_by_symbol(lags_acf)).intersection(
            set(common_lags_by_symbol(lags_pacf))
        )
    )

    moving_averages_list = [13, 50, 100, 200]
    on_cols = [f"close{type_scale_suffix}", "close"]

    df = calculate_rolling_avgs_by_symbol(df, moving_averages_list, on_cols)
    df = df.drop_nulls()
    df = extract_date_features(df, "date")

    # CALCULATING CONTINUOUS RED/GREEN CANDLES
    df = df.with_columns(
        pl.when(pl.col("open") > pl.col("close"))
        .then(1)
        .otherwise(0)
        .alias("green_candle"),
        pl.when(pl.col("open") <= pl.col("close"))
        .then(1)
        .otherwise(0)
        .alias("red_candle"),
    )

    df = df.with_columns(
        pl.when(pl.col("green_candle") == 1)
        .then(
            pl.col("green_candle")
            .cum_sum()
            .over(
                (
                    pl.col("green_candle") != pl.col("green_candle").shift(1)
                ).cum_sum()
            )
            - 1
        )
        .otherwise(0)
        .alias("cons_green_candles"),
        pl.when(pl.col("red_candle") == 1)
        .then(
            pl.col("red_candle")
            .cum_sum()
            .over(
                (
                    pl.col("red_candle") != pl.col("red_candle").shift(1)
                ).cum_sum()
            )
            - 1
        )
        .otherwise(0)
        .alias("cons_red_candles"),
    )

    other_lags = [1, 2, 3, 4, 6, 12, 72]
    all_lags = list(set(common_lags).union(set(other_lags)))

    print("Calculate RSI...")
    df = calculate_rsi_by_symbol(df, window=14)
    print("Calculate MACD...")
    df = calculate_macd_by_symbol(df)
    print("Calculate Bollinger Bands...")
    df = calculate_bollinger_bands_by_symbol(df)
    print("Scaling Features...")
    df = scale_features(df, ["upper_band", "lower_band"])
    print("Calculate Lagged Features...")
    df = create_lagged_features_by_symbol(
        df=df,
        feature_col=f"close{type_scale_suffix}",
        significant_lags=all_lags,
    )
    df = df.drop_nulls()

    symbols_list = [
        "ETH", "BTC", "SOL",
        "XRP", "DOGE", "TRX",
        "ADA", "LTC", "ATOM"
    ]

    symbol_encoding = {
        symbol: i for i, symbol in enumerate(symbols_list)
    }

    df = df.with_columns(
        pl.col("symbol")
        .replace(symbol_encoding)
        .cast(pl.Int32)
        .alias("symbol_encoded")
    )

    data_sp = get_all_data(START_DATE, END_DATE)
    data_sp = data_sp.with_columns(
        pl.col("date")
        .dt.truncate("1h")
        .dt.replace_time_zone(None)
        .dt.cast_time_unit("us")
        .alias("date"),
        pl.col("SP500").fill_null(strategy="forward").alias("SP500"),
        pl.col("NASDAQ").fill_null(strategy="forward").alias("NASDAQ"),
    ).sort("date")
    data_sp = scale_features(
        data_sp,
        ["SP500", "NASDAQ"],
        type_scale="pct_change",
        suffix="_pct_change",
    )
    data_sp = data_sp.drop_nulls()

    data_vix = get_vix_data(START_DATE, END_DATE)
    data_vix = data_vix.with_columns(
        pl.col("date")
        .dt.truncate("1h")
        .dt.replace_time_zone(None)
        .dt.cast_time_unit("us")
        .alias("date"),
        pl.col("VIX").fill_null(strategy="forward").alias("VIX"),
    ).sort("date")
    data_vix = scale_features(
        data_vix, ["VIX"], type_scale="pct_change", suffix="_pct_change"
    )

    data_usd = get_stock_data("DX-Y.NYB", START_DATE, END_DATE)
    data_usd = data_usd.with_columns(
        pl.col("date")
        .dt.truncate("1h")
        .dt.convert_time_zone("UTC")
        .dt.replace_time_zone(None)
        .dt.cast_time_unit("us")
        .alias("date"),
        pl.col("Adj Close").fill_null(strategy="forward").alias("DXY"),
    ).sort("date")
    data_usd = scale_features(
        data_usd, ["DXY"], type_scale="pct_change", suffix="_pct_change"
    )

    all_dates = pl.DataFrame(
        pl.datetime_range(
            START_DATE,
            END_DATE,
            dt.timedelta(hours=1),
            time_unit="us",
            eager=True,
        ).alias("date")
    )
    data = all_dates.join(data_sp, on="date", how="left").with_columns(
        pl.col("SP500_pct_change").fill_null(strategy="forward"),
        pl.col("NASDAQ_pct_change").fill_null(strategy="forward"),
    )
    data = (
        data.join(data_vix, on="date", how="left")
        .with_columns(
            pl.col("VIX_pct_change").fill_null(strategy="forward"),
        )
        .sort("date")
    )
    data = (
        data.join(data_usd, on="date", how="left")
        .with_columns(
            pl.col("DXY_pct_change").fill_null(strategy="forward"),
        )
        .sort("date")
    )

    ##########################################################################
    # shifto perche suppongo che l'ora prima
    # influenzi i dati crypto dell'ora successiva
    ##########################################################################
    data = data.with_columns(
        pl.col("SP500_pct_change").shift().alias("SP500_pct_change"),
        pl.col("NASDAQ_pct_change").shift().alias("NASDAQ_pct_change"),
        pl.col("VIX_pct_change").shift().alias("VIX_pct_change"),
        pl.col("DXY_pct_change").shift().alias("DXY_pct_change"),
    )

    data = data[
        [
            "date",
            "SP500_pct_change",
            "NASDAQ_pct_change",
            "VIX_pct_change",
            "DXY_pct_change",
        ]
    ].drop_nulls()
    df_merged = df.join(data, on="date", how="left")

    assert (
        len(
            df_merged.filter(pl.col("SP500_pct_change").is_null()).filter(
                pl.col("symbol") == "BTC"
            )
        )
        == 0
    )
    assert (
        len(
            df_merged.filter(pl.col("VIX_pct_change").is_null()).filter(
                pl.col("symbol") == "BTC"
            )
        )
        == 0
    )

    macro_data = get_macroeconomic_data(START_DATE, END_DATE)
    all_macro_data = all_dates.join(
        macro_data, on="date", how="left"
    ).with_columns(
        pl.col("US_FED_RATE")
        .fill_null(strategy="forward")
        .alias("US_FED_RATE"),
        pl.col("US_INFLATION")
        .fill_null(strategy="forward")
        .alias("US_INFLATION"),
        pl.col("US_UNEMPLOYMENT")
        .fill_null(strategy="forward")
        .alias("US_UNEMPLOYMENT"),
        pl.col("US_GDP").fill_null(strategy="forward").alias("US_GDP"),
    )
    df_merged = df_merged.join(all_macro_data, on="date", how="left")

    df_merged.write_parquet(DATA_PATH / "feature_engineering.parquet")
