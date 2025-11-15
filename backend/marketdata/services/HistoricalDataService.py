import time
import math
from pathlib import Path

import requests
import pandas as pd
from django.conf import settings

MS_IN_DAY = 24 * 60 * 60 * 1000
DAYS_IN_YEAR = 365

COLS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "num_trades",
    "taker_buy_base", "taker_buy_quote", "ignore"
]

FLOAT_COLS = [
    "open", "high", "low", "close", "volume",
    "quote_volume", "taker_buy_base", "taker_buy_quote"
]

SYMBOLS = ["BTCUSDC", "ETHUSDC", "BNBUSDC", "XRPUSDC"]
SYMBOL_SUFFIXES = {
    "BTCUSDC": "btc",
    "ETHUSDC": "eth",
    "BNBUSDC": "bnb",
    "XRPUSDC": "xrp",
}


def _fetch(url):
    while True:
        r = requests.get(url, timeout=30)
        if r.status_code == 200:
            return r.json()
        if r.status_code == 429:
            retry = float(r.headers.get("Retry-After", "5"))
            time.sleep(retry)
            continue
        return None

def _apply_exclusions(df, excluded_cols = None):
    if not excluded_cols:
        return df
    cols_to_drop = [
        col for col in excluded_cols
        if col in df.columns and col != "open_time"
    ]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
    return df


def fetch_multi_symbol_df_excluding(
    years_back = 10,
    excluded_cols = None,
):
    df = fetch_multi_symbol_df(years_back=years_back)
    df = _apply_exclusions(df, excluded_cols)
    return df

def fetch_historical_df(symbol, years_back = 10, end_time_ms = None):
    if end_time_ms is None:
        end_time_ms = int(time.time() * 1000)
    total_days = years_back * DAYS_IN_YEAR
    batch_size_days = 1000
    batches = math.ceil(total_days / batch_size_days)
    dfs = []
    for i in range(batches):
        batch_end_ms = end_time_ms - i * batch_size_days * MS_IN_DAY
        url = (
            "https://www.binance.com/api/v3/uiKlines"
            f"?symbol={symbol}&interval=1d&limit=1000&endTime={batch_end_ms}"
        )
        print(f"[{symbol}] Pobieranie partii {i + 1}/{batches}... endTime={batch_end_ms}")
        data = _fetch(url)
        if not data:
            print(f"[{symbol}] Brak danych / błąd, lecimy dalej.")
            continue
        df = pd.DataFrame(data, columns=COLS)
        df[FLOAT_COLS] = df[FLOAT_COLS].astype(float)
        df["num_trades"] = df["num_trades"].astype(int)
        df = df.drop(columns=["ignore"])
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
        df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")
        dfs.append(df)
    if not dfs:
        raise ValueError(f"Nie udało się pobrać żadnych danych z Binance dla symbolu {symbol}.")
    full = pd.concat(dfs, ignore_index=True).sort_values("open_time")
    cutoff_start_ms = end_time_ms - total_days * MS_IN_DAY
    cutoff_start = pd.to_datetime(cutoff_start_ms, unit="ms")
    full = full[full["open_time"] >= cutoff_start]
    return full


def fetch_multi_symbol_df(years_back = 10):
    end_time_ms = int(time.time() * 1000)
    merged_df = None
    for symbol in SYMBOLS:
        suffix = SYMBOL_SUFFIXES[symbol]
        df = fetch_historical_df(symbol=symbol, years_back=years_back, end_time_ms=end_time_ms)
        df_trimmed = df[["open_time", "open", "high", "low", "close", "volume", "num_trades"]].copy()
        df_trimmed = df_trimmed.rename(
            columns={
                "open": f"open_{suffix}",
                "high": f"high_{suffix}",
                "low": f"low_{suffix}",
                "close": f"close_{suffix}",
                "volume": f"volume_{suffix}",
                "num_trades": f"num_trades_{suffix}",
            }
        )
        if merged_df is None:
            merged_df = df_trimmed
        else:
            merged_df = merged_df.merge(df_trimmed, on="open_time", how="inner")
    if merged_df is None:
        raise ValueError("Nie udało się zbudować df dla żadnego symbolu.")
    merged_df = merged_df.sort_values("open_time").reset_index(drop=True)
    desired_cols = [
        "open_time",
        "open_btc", "high_btc", "close_btc", "low_btc",
        "open_eth", "high_eth", "close_eth", "low_eth",
        "open_bnb", "high_bnb", "close_bnb", "low_bnb",
        "open_xrp", "high_xrp", "close_xrp", "low_xrp",
        "volume_btc", "volume_xrp", "volume_bnb", "volume_eth",
        "num_trades_btc", "num_trades_bnb", "num_trades_eth", "num_trades_xrp",
    ]
    merged_df = merged_df[desired_cols]
    return merged_df


def generate_parquet_file(
    years_back = 10,
    filename = "data.parquet",
    excluded_cols = None,
):
    full_df = fetch_multi_symbol_df(years_back=years_back)
    df_to_save = _apply_exclusions(full_df.copy(), excluded_cols)
    output_dir = Path(settings.BASE_DIR) / "data"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / filename
    df_to_save.to_parquet(output_path, index=False)
    return {
        "rows": int(len(df_to_save)),
        "from": full_df["open_time"].min().isoformat(),
        "to": full_df["open_time"].max().isoformat(),
        "file": str(output_path),
        "years_back": years_back,
        "symbols": SYMBOLS,
    }