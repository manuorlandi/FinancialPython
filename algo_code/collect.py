import sys
import os
import time
import polars as pl
import datetime as dt
from pathlib import Path
from pybit.unified_trading import HTTP
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fin_utilities

print(os.getcwd())
cfg = fin_utilities.__cfg_reading("pred")


PROJECT_DIR = eval(cfg["PROJECT_PATH"])
DATA_PATH = PROJECT_DIR / cfg["DATA_FOLDER"]

BYBIT_API_KEY = "5u0HfwB5UPJeiQo3WR"
BYBIT_SECRET_KEY = "hjFn5aEvyuVEZ1dnna6R4s1NS1vw3vZdJFIL"


def get_tickers(session: HTTP):
    result = session.get_tickers(
        category="linear"
    ).get("result")["list"]

    tickers = [
        asset["symbol"]
        for asset in result if asset["symbol"].endswith("USDT")
    ]
    return tickers


def get_last_timestamp(df: pl.DataFrame):
    return df.get_column("timestamp")[-1]


def format_data(response):
    data = response.get("list", None)
    if not data:
        return

    data = pl.DataFrame(data, schema={
        "timestamp": pl.Int64,
        "open": pl.Float64,
        "high": pl.Float64,
        "low": pl.Float64,
        "close": pl.Float64,
        "volume": pl.Float64,
        "turnover": pl.Float64,
    }, orient="row")
    # Reverse the DataFrame
    data = data.reverse()
    # Convert timestamp to datetime
    data = data.with_columns(
        (pl.col("timestamp") * 1000).cast(pl.Datetime).alias("date")
    )

    return data


def get_data_from(
    session: HTTP,
    category: str,
    symbol: str,
    start: int,
    end: int = None,
    interval: int = 60,
):
    df = pl.DataFrame()
    while True:
        response = session.get_kline(
            category=category,
            symbol=symbol,
            start=start,
            interval=interval
        ).get("result")

        latest = format_data(response)
        if latest is None or latest.shape[0] == 0:
            break

        latest_timestamp = latest.select(
            pl.col("timestamp").cast(pl.Int64)
        ).to_series().max()

        if end is not None and latest_timestamp > end:
            df = df.filter(pl.col("timestamp").cast(pl.Int64) <= int(end))
            break

        start = get_last_timestamp(latest)
        time.sleep(0.1)

        df = pl.concat([df, latest])
        print(
            "Collecting data starting",
            f"{dt.datetime.fromtimestamp(start / 1000)}"
        )
        if len(latest) == 1:
            break

    df = df.unique(subset=["timestamp"], keep="last")

    return df.sort("date")


def return_index_if_exists(df_series, curr_idx, val, pos_crit, max_length):
    # Slicing the series from curr_idx + 1 to curr_idx + 1 + max_length
    future_series = df_series.slice(curr_idx + 1, max_length)
    # display(future_series)
    if pos_crit:
        indices = future_series.filter(future_series >= val).arg_min()
    else:
        indices = future_series.filter(future_series <= val).arg_min()

    # Return the relative index if condition met within max_length,
    # otherwise return max_length
    if indices is not None and indices < max_length:
        return indices + 1  # +1 because of slicing offset
    else:
        return max_length


def labelize_output_according_criterion2(
    df,
    trade="long",
    threshold=0.01,
    risk_reward_ratio=0.5,
    max_trade_length=6,
    wrt="close",
    hl=["high", "low"],
):

    if trade == "long":
        df_ = df.with_columns([
            (pl.col(wrt) * (1 + threshold / risk_reward_ratio)).alias("TP"),
            (pl.col(wrt) * (1 - threshold)).alias("SL"),

        ])
        pos_crit = True
        neg_crit = False
    elif trade == "short":
        df_ = df.with_columns([
            (pl.col(wrt) * (1 - threshold / risk_reward_ratio)).alias("TP"),
            (pl.col(wrt) * (1 + threshold)).alias("SL"),

        ])
        pos_crit = False
        neg_crit = True

    min_above = []
    min_below = []

    for idx in range(len(df_)):
        # display(df_.slice(idx,1))
        TP = df_["TP"][idx]
        SL = df_["SL"][idx]

        if idx != len(df_) - 1:
            candidates_above_minima = [
                return_index_if_exists(
                    df_series=df_[v],
                    curr_idx=idx,
                    val=TP,
                    pos_crit=pos_crit,
                    max_length=max_trade_length
                )
                for v in hl
            ]
            candidates_below_minima = [
                return_index_if_exists(
                    df_series=df_[v],
                    curr_idx=idx,
                    val=SL,
                    pos_crit=neg_crit,
                    max_length=max_trade_length
                )
                for v in hl
            ]

            min_above.append(min(candidates_above_minima))
            min_below.append(min(candidates_below_minima))

        else:
            min_above.append(None)
            min_below.append(None)

    df_ = df_.with_columns(pl.Series("min_above", min_above))
    df_ = df_.with_columns(pl.Series("min_below", min_below))

    # Generate the signal
    signal = (
        (df_["min_above"] < df_["min_below"]) &
        (df_["min_above"] <= max_trade_length)
    ).cast(pl.Int8) * (1 if trade == "long" else -1)
    return signal


if __name__ == "__main__":
    # https://bybit-exchange.github.io/docs/api-explorer/v5/market/kline
    start = int(dt.datetime(2023, 1, 1).timestamp() * 1000)
    interval = 60
    symbols_list = [
        "ETHUSDT", "BTCUSDT", "SOLUSDT",
        "XRPUSDT", "DOGEUSDT", "TRXUSDT",
        "ADAUSDT", "LTCUSDT", "ATOMUSDT"
    ]

    session = HTTP(
        testnet=False,
        api_key=BYBIT_API_KEY,
        api_secret=BYBIT_SECRET_KEY,
    )
    server_time = session.get_server_time()["time"] / 1000
    print(
        "Current server time:",
        dt.datetime.fromtimestamp(server_time).strftime("%Y-%m-%d %H:%M:%S")
    )

    df_orig = pl.DataFrame()
    for symbol in symbols_list:
        print(f"Collecting data for {symbol}")
        tmp_df = get_data_from(
            session=session,
            category="spot",
            symbol=symbol,
            start=start,
            end=None,
            interval=interval,
        )
        tmp_df = tmp_df.with_columns(
            pl.lit(symbol.replace("USDT", "")).alias("symbol")
        )
        tmp_df = tmp_df.with_columns(
            labelize_output_according_criterion2(tmp_df, "long")
            .alias("long_signal")
        )
        tmp_df = tmp_df.with_columns(
            labelize_output_according_criterion2(tmp_df, "short")
            .alias("short_signal")
        )

        df_orig = df_orig.vstack(tmp_df)

    df_orig.write_parquet(DATA_PATH / "bybit_data.parquet")
