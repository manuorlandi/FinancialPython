import os
import sys
import datetime as dt
from pathlib import Path
from pybit.unified_trading import HTTP
from fredapi import Fred
print(str(os.getcwd()))


PROJECT_ROOT = Path(__file__).parent.parent

DATA_PATH = PROJECT_ROOT / "data"

INTERVAL_MAPPING = {
    1: '1m',
    3: '3m',
    5: '5m',
    15: '15m',
    30: '30m',
    60: '1h',
    120: '2h',
    240: '4h',
    360: '6h',
    720: '12h',
    1440: 'D',
    10080: 'W',
    43200: 'M'
}

BYBIT_API_KEY = "5u0HfwB5UPJeiQo3WR"
BYBIT_SECRET_KEY = "hjFn5aEvyuVEZ1dnna6R4s1NS1vw3vZdJFIL"

SESSION = HTTP(
    testnet=False,
    api_key=BYBIT_API_KEY,
    api_secret=BYBIT_SECRET_KEY,
)

FRED_SESSION = Fred(api_key="a2bdf7d81b8f0362f82e976bb92fbdb4")

server_time = SESSION.get_server_time()["time"] / 1000
print(
    "Current server time:",
    dt.datetime.fromtimestamp(server_time).strftime("%Y-%m-%d %H:%M:%S")
)

INTERVAL = 60
STR_INTERVAL = INTERVAL_MAPPING[INTERVAL]

SYMBOLS_LIST = [
    "ETHUSDT", "BTCUSDT", "SOLUSDT",
    "XRPUSDT", "DOGEUSDT", "TRXUSDT",
    "ADAUSDT", "LTCUSDT", "ATOMUSDT"
]
