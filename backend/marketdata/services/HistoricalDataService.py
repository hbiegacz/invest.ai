import os
import time
import math
from pathlib import Path
import io

import requests
import pandas as pd
from django.conf import settings


class HistoricalDataService:
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
    STOOQ_SPX_URL = "https://stooq.pl/q/d/l/?s=^spx&i=d"
    FRED_SERIES_URL = "https://api.stlouisfed.org/fred/series/observations"

    def __init__(self, years_back=10, timeout=30):
        self.years_back = years_back
        self.timeout = timeout

    def _fetch(self, url: str):
        while True:
            r = requests.get(url, timeout=self.timeout)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 429:
                retry = float(r.headers.get("Retry-After", "5"))
                time.sleep(retry)
                continue
            return None

    def _apply_exclusions(self, df, excluded_cols=None):
        if not excluded_cols:
            return df
        cols_to_drop = [
            col for col in excluded_cols
            if col in df.columns and col != "open_time"
        ]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
        return df

    def fetch_historical_df(
        self,
        symbol,
        years_back=None,
        end_time_ms=None,
    ):
        if years_back is None:
            years_back = self.years_back
        if end_time_ms is None:
            end_time_ms = int(time.time() * 1000)
        total_days = years_back * self.DAYS_IN_YEAR
        batch_size_days = 1000
        batches = math.ceil(total_days / batch_size_days)
        dfs = []
        for i in range(batches):
            batch_end_ms = end_time_ms - i * batch_size_days * self.MS_IN_DAY
            url = (
                "https://www.binance.com/api/v3/uiKlines"
                f"?symbol={symbol}&interval=1d&limit=1000&endTime={batch_end_ms}"
            )
            print(f"[{symbol}] Pobieranie partii {i + 1}/{batches}... endTime={batch_end_ms}")
            data = self._fetch(url)
            if not data:
                print(f"[{symbol}] Brak danych / błąd, lecimy dalej.")
                continue
            df = pd.DataFrame(data, columns=self.COLS)
            df[self.FLOAT_COLS] = df[self.FLOAT_COLS].astype(float)
            df["num_trades"] = df["num_trades"].astype(int)
            df = df.drop(columns=["ignore"])
            df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
            df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")
            dfs.append(df)
        if not dfs:
            raise ValueError(f"Nie udało się pobrać żadnych danych z Binance dla symbolu {symbol}.")
        full = pd.concat(dfs, ignore_index=True).sort_values("open_time")
        cutoff_start_ms = end_time_ms - total_days * self.MS_IN_DAY
        cutoff_start = pd.to_datetime(cutoff_start_ms, unit="ms")
        full = full[full["open_time"] >= cutoff_start]
        full = full.drop_duplicates(subset=["open_time"], keep="last").reset_index(drop=True)
        return full

    def fetch_spx_df(self):
        resp = requests.get(self.STOOQ_SPX_URL, timeout=self.timeout)
        resp.raise_for_status()
        buf = io.StringIO(resp.text)
        df = pd.read_csv(buf)
        df = df.rename(
            columns={
                "Data": "open_time",
                "Otwarcie": "open_spx",
                "Najwyzszy": "high_spx",
                "Najnizszy": "low_spx",
                "Zamkniecie": "close_spx",
                "Wolumen": "volume_spx",
            }
        )
        df["open_time"] = pd.to_datetime(df["open_time"])
        df = df.sort_values("open_time").reset_index(drop=True)
        return df

    def _fetch_fred_series(self, series_id, observation_start="2018-01-01"):
        api_key = os.getenv("FRED_API_KEY")
        if not api_key:
            raise ValueError("FRED_API_KEY is not set in environment variables")
        params = {
            "series_id": series_id,
            "observation_start": observation_start,
            "sort_order": "asc",
            "api_key": api_key,
            "file_type": "json",
        }
        resp = requests.get(self.FRED_SERIES_URL, params=params, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def _fetch_gdp_df(self):
        data = self._fetch_fred_series("GDP", observation_start="2018-01-01")
        observations = data.get("observations", [])
        if not observations:
            return pd.DataFrame(columns=["open_time", "gdp"])
        df = pd.DataFrame(observations)[["date", "value"]]
        df = df.rename(columns={"date": "open_time", "value": "gdp"})
        df["open_time"] = pd.to_datetime(df["open_time"])
        df["gdp"] = pd.to_numeric(df["gdp"], errors="coerce")
        df = df.sort_values("open_time").reset_index(drop=True)
        return df

    def _fetch_unrate_df(self):
        data = self._fetch_fred_series("UNRATE", observation_start="2018-01-01")
        observations = data.get("observations", [])
        if not observations:
            return pd.DataFrame(columns=["open_time", "unrate"])
        df = pd.DataFrame(observations)[["date", "value"]]
        df = df.rename(columns={"date": "open_time", "value": "unrate"})
        df["open_time"] = pd.to_datetime(df["open_time"])
        df["unrate"] = pd.to_numeric(df["unrate"], errors="coerce")
        df = df.sort_values("open_time").reset_index(drop=True)
        return df

    def fetch_multi_symbol_df(self, years_back=None):
        if years_back is None:
            years_back = self.years_back
        end_time_ms = int(time.time() * 1000)
        merged_df = None
        for symbol in self.SYMBOLS:
            suffix = self.SYMBOL_SUFFIXES[symbol]
            df = self.fetch_historical_df(
                symbol=symbol,
                years_back=years_back,
                end_time_ms=end_time_ms,
            )
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
        merged_df = (
            merged_df
            .sort_values("open_time")
            .drop_duplicates(subset=["open_time"], keep="last")
            .reset_index(drop=True)
        )
        spx_df = self.fetch_spx_df()
        min_time = merged_df["open_time"].min()
        max_time = merged_df["open_time"].max()
        spx_df = spx_df[
            (spx_df["open_time"] >= min_time)
            & (spx_df["open_time"] <= max_time)
        ]
        merged_df = merged_df.merge(spx_df, on="open_time", how="left")
        gdp_df = self._fetch_gdp_df()
        unrate_df = self._fetch_unrate_df()
        if not gdp_df.empty:
            gdp_df = gdp_df[
                (gdp_df["open_time"] >= min_time)
                & (gdp_df["open_time"] <= max_time)
            ]
            merged_df = merged_df.merge(gdp_df, on="open_time", how="left")
        if not unrate_df.empty:
            unrate_df = unrate_df[
                (unrate_df["open_time"] >= min_time)
                & (unrate_df["open_time"] <= max_time)
            ]
            merged_df = merged_df.merge(unrate_df, on="open_time", how="left")
        merged_df = merged_df.sort_values("open_time").reset_index(drop=True)
        cols_to_ffill = [
            "gdp",
            "unrate",
            "open_spx",
            "high_spx",
            "close_spx",
            "low_spx",
            "volume_spx",
        ]
        cols_to_ffill = [c for c in cols_to_ffill if c in merged_df.columns]
        if cols_to_ffill:
            merged_df[cols_to_ffill] = (
                merged_df[cols_to_ffill]
                .ffill()
                .bfill()
            )
        desired_cols = [
            "open_time",
            "open_btc", "high_btc", "close_btc", "low_btc",
            "open_eth", "high_eth", "close_eth", "low_eth",
            "open_bnb", "high_bnb", "close_bnb", "low_bnb",
            "open_xrp", "high_xrp", "close_xrp", "low_xrp",
            "volume_btc", "volume_xrp", "volume_bnb", "volume_eth",
            "num_trades_btc", "num_trades_bnb", "num_trades_eth", "num_trades_xrp",
            "open_spx", "high_spx", "close_spx", "low_spx", "volume_spx",
            "gdp", "unrate",
        ]
        existing_cols = [c for c in desired_cols if c in merged_df.columns]
        merged_df = merged_df[existing_cols]
        return merged_df

    def fetch_multi_symbol_df_excluding(
        self,
        years_back=None,
        excluded_cols=None,
    ):
        full_df = self.fetch_multi_symbol_df(years_back=years_back)
        df = self._apply_exclusions(full_df, excluded_cols)
        return df

    def generate_parquet_file(
        self,
        years_back=None,
        filename="data.parquet",
        excluded_cols=None,
    ):
        full_df = self.fetch_multi_symbol_df(years_back=years_back)
        df_to_save = self._apply_exclusions(full_df.copy(), excluded_cols)
        output_dir = Path(settings.BASE_DIR) / "data"
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / filename
        df_to_save.to_parquet(output_path, index=False)
        return {
            "rows": int(len(df_to_save)),
            "from": full_df["open_time"].min().isoformat(),
            "to": full_df["open_time"].max().isoformat(),
            "file": str(output_path),
            "years_back": years_back if years_back is not None else self.years_back,
            "symbols": self.SYMBOLS,
        }
