#!/usr/bin/env python3
# binance_ui_klines.py
import sys
import json
import time
import csv
from datetime import datetime
from typing import List, Any, Optional
import requests

BASE_URL = "https://www.binance.com/api/v3/uiKlines"

def ms_to_iso(ms: int) -> str:
    return datetime.utcfromtimestamp(ms / 1000).strftime("%Y-%m-%d %H:%M:%S")

def fetch_ui_klines(
    symbol: str = "BTCUSDC",
    interval: str = "4h",
    limit: int = 10000,
    end_time: Optional[int] = 1760777000000,
    start_time: Optional[int] = None,
    timeout: int = 20
) -> List[List[Any]]:
    """
    Zwraca listę świec (tablic), dokładnie tak jak zwraca endpoint uiKlines.
    """
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit
    }
    if end_time is not None:
        params["endTime"] = end_time
    if start_time is not None:
        params["startTime"] = start_time

    headers = {
        # Niektóre CDN-y lubią mieć UA:
        "User-Agent": "python-requests uiKlines fetcher/1.0"
    }
    r = requests.get(BASE_URL, params=params, headers=headers, timeout=timeout)
    # Obsługa HTTP-error:
    try:
        r.raise_for_status()
    except requests.HTTPError as e:
        # Spróbujmy pokazać sensowny komunikat JSON z błędu:
        try:
            err = r.json()
        except Exception:
            err = r.text
        raise SystemExit(f"HTTP {r.status_code}: {err}") from e

    # Parsowanie JSON
    try:
        data = r.json()
    except ValueError:
        raise SystemExit(f"Niepoprawny JSON: {r.text[:300]}")

    if not isinstance(data, list):
        # Binance zwykle zwraca listę list. Jeśli nie, pokażmy “raw”:
        raise SystemExit(f"Nieoczekiwany format odpowiedzi:\n{json.dumps(data, indent=2)[:1000]}")

    return data

def to_csv(rows: List[List[Any]], path: str) -> None:
    """
    uiKlines zwykle zwraca (jak /klines):
    [ openTime, open, high, low, close, volume, closeTime, quoteAssetVolume,
      numberOfTrades, takerBuyBaseAssetVolume, takerBuyQuoteAssetVolume, ignore ]
    Zapisujemy nagłówki defensywnie (gdyby kolumn było mniej/więcej).
    """
    # Przygotuj nagłówki “domyślne”
    default_headers = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base", "taker_buy_quote", "ignore"
    ]
    max_len = max(len(r) for r in rows) if rows else 0
    headers = default_headers[:max_len] if max_len <= len(default_headers) else [f"col_{i}" for i in range(max_len)]

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(headers)
        for r in rows:
            w.writerow(r + [None] * (max_len - len(r)))  # wyrównanie

def main():
    # Parametry jak w Twoim przykładzie:
    symbol = "BTCUSDC"
    interval = "5m"
    limit = 10000
    end_time = 1760778967066  # ms

    rows = fetch_ui_klines(symbol=symbol, interval=interval, limit=limit, end_time=end_time)

    print(f"Pobrano {len(rows)} świec dla {symbol} ({interval}).")
    if rows:
        # Pokażmy pierwszą i ostatnią wiersz, ładnie:
        first = rows[0]
        last = rows[-1]

        def fmt_row(row):
            # Jeśli są przynajmniej open_time i close_time — pokaż ISO:
            ot = ms_to_iso(row[0]) if len(row) > 0 and isinstance(row[0], (int, float)) else "n/d"
            ct = ms_to_iso(row[6]) if len(row) > 6 and isinstance(row[6], (int, float)) else "n/d"
            o = row[1] if len(row) > 1 else "n/d"
            h = row[2] if len(row) > 2 else "n/d"
            l = row[3] if len(row) > 3 else "n/d"
            c = row[4] if len(row) > 4 else "n/d"
            return f"open_time={ot}, O={o}, H={h}, L={l}, C={c}, close_time={ct}"

        print("Pierwsza świeca: ", fmt_row(first))
        print("Ostatnia świeca:  ", fmt_row(last))

    # Zapis do CSV (opcjonalnie)
    out = f"binance_{symbol}_{interval}.csv"
    to_csv(rows, out)
    print(f"Zapisano CSV: {out}")

if __name__ == "__main__":
    main()
