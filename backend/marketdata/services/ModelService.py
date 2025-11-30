import os
import pandas as pd
from joblib import load
from django.conf import settings
from pathlib import Path

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

    def linear_regression(self, force_refresh: bool = False):
        self._ensure_fresh_file_if_needed(force_refresh=force_refresh)
        model_path = Path(settings.BASE_DIR) / "data" / "linear_regression_btc.pkl"
        if not model_path.exists():
            raise FileNotFoundError(f"Trained model file not found at: {model_path}")
        payload = load(model_path)
        pipeline = payload["model"]
        if "features" not in payload:
            raise ValueError("Trained model payload missing 'features' list.")
        scaler = pipeline.named_steps.get("scaler")
        if scaler is None:
            raise ValueError("Pipeline does not contain a 'scaler' step named 'scaler'.")
        model_features = list(getattr(scaler, "feature_names_in_", payload["features"]))
        cols_to_read = list(dict.fromkeys(model_features + ["open_time", "close_btc"]))
        df = pd.read_parquet(
            self.data_reader.file_path,
            columns=cols_to_read,
            engine="pyarrow",
        )
        if df.empty:
            raise ValueError("No data available in historical_data.parquet for linear regression model.")
        missing = [c for c in model_features if c not in df.columns]
        if missing:
            raise ValueError(f"Missing feature columns in parquet: {missing}")
        df = df.sort_values("open_time").reset_index(drop=True)
        last_row_df = df.iloc[[-1]]
        X_last = last_row_df[model_features].astype(float).to_numpy()
        last_close = float(last_row_df["close_btc"].iloc[0])
        ret_pred = float(pipeline.predict(X_last)[0])
        predicted_close = last_close * (1.0 + ret_pred)
        return predicted_close