import os
import pandas as pd

from .DataReaderService import DataReaderService

class ModelService:
    def __init__(self):
        self.data_reader = DataReaderService()

    def _ensure_fresh_file_if_needed(self, force_refresh=False):
        if force_refresh or not os.path.exists(self.data_reader.file_path):
            self.data_reader._refresh_data_file()

    def naive(self, force_refresh=False):
        self._ensure_fresh_file_if_needed(force_refresh=force_refresh)
        try:
            df = pd.read_parquet(
                self.data_reader.file_path,
                columns=["close_btc"],
                engine="pyarrow",
            )
        except FileNotFoundError:
            self.data_reader._refresh_data_file()
            df = pd.read_parquet(
                self.data_reader.file_path,
                columns=["close_btc"],
                engine="pyarrow",
            )
        if df.empty:
            raise ValueError("No data available in historical_data.parquet for naive BTC model.")
        if "close_btc" not in df.columns:
            raise ValueError("Column 'close_btc' not found in historical_data.parquet.")
        last_close = df["close_btc"].iloc[-1]
        return float(last_close)
