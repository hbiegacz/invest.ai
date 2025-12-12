import os
import pandas as pd
import math
import numpy as np
import re
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
    
    def _rf_base_columns(self):
        assets_all = ["btc", "eth", "bnb", "xrp", "spx"]
        assets_crypto = ["btc", "eth", "bnb", "xrp"]
        cols = ["open_time", "gdp", "unrate"]
        for a in assets_all:
            cols += [f"high_{a}", f"low_{a}", f"close_{a}", f"volume_{a}"]
        for a in assets_crypto:
            cols += [f"num_trades_{a}"]
        return list(dict.fromkeys(cols))

    def _parse_rf_spans_windows(self, feature_names):
        spans = set()
        wins = set()
        span_re = re.compile(r"_s(\d+)$")
        win_re = re.compile(r"_w(\d+)$")
        for f in feature_names:
            m = span_re.search(f)
            if m:
                spans.add(int(m.group(1)))
            m = win_re.search(f)
            if m:
                wins.add(int(m.group(1)))
        return sorted(spans), sorted(wins)

    def _compute_rf_features(self, df, spans, vol_windows):
        df = df.sort_values("open_time").reset_index(drop=True)
        assets_all = ["btc", "eth", "bnb", "xrp", "spx"]
        assets_crypto = ["btc", "eth", "bnb", "xrp"]
        for a in assets_all:
            df[f"hl2_{a}"] = (df[f"low_{a}"] + df[f"high_{a}"]) / 2.0
        df["volume_sum"] = df[[f"volume_{a}" for a in assets_all]].sum(axis=1)
        df["num_trades_sum"] = df[[f"num_trades_{a}" for a in assets_crypto]].sum(axis=1)
        for a in assets_all:
            df[f"ret_close_{a}"] = np.log(df[f"close_{a}"]).diff()
            df[f"ret_hl2_{a}"] = np.log(df[f"hl2_{a}"]).diff()
        df["dlog_volume_sum"] = np.log1p(df["volume_sum"]).diff()
        df["dlog_num_trades_sum"] = np.log1p(df["num_trades_sum"]).diff()
        for span in spans:
            for a in assets_all:
                df[f"ewm_ret_close_{a}_s{span}"] = df[f"ret_close_{a}"].shift(1).ewm(span=span, adjust=False).mean()
                df[f"ewm_ret_hl2_{a}_s{span}"] = df[f"ret_hl2_{a}"].shift(1).ewm(span=span, adjust=False).mean()
            df[f"ewm_dlog_volume_sum_s{span}"] = df["dlog_volume_sum"].shift(1).ewm(span=span, adjust=False).mean()
            df[f"ewm_dlog_num_trades_sum_s{span}"] = df["dlog_num_trades_sum"].shift(1).ewm(span=span, adjust=False).mean()
        for w in vol_windows:
            df[f"roll_std_ret_close_btc_w{w}"] = df["ret_close_btc"].shift(1).rolling(window=w, min_periods=w).std()
        df["gdp_lag1"] = df["gdp"].shift(1)
        df["unrate_lag1"] = df["unrate"].shift(1)
        df["gdp_growth"] = df["gdp_lag1"].pct_change()
        df["unrate_change"] = df["unrate_lag1"] - df["unrate_lag1"].shift(1)
        return df

    def random_forest(self, force_refresh=False):
        self._ensure_fresh_file_if_needed(force_refresh=force_refresh)
        model_path = Path(settings.BASE_DIR) / "data" / "random_forest_btc.pkl"
        if not model_path.exists():
            raise FileNotFoundError(f"Trained RF model file not found at: {model_path}")
        payload = load(model_path)
        model = payload.get("model")
        feature_names = payload.get("features")
        if model is None or not feature_names:
            raise ValueError("RF payload missing 'model' or 'features'.")
        spans, vol_windows = self._parse_rf_spans_windows(feature_names)
        df = pd.read_parquet(
            self.data_reader.file_path,
            columns=self._rf_base_columns(),
            engine="pyarrow",
        )
        if df.empty:
            raise ValueError("No data available in historical_data.parquet for random forest model.")
        df_feat = self._compute_rf_features(df, spans=spans, vol_windows=vol_windows)
        last = df_feat.iloc[[-1]]
        missing = [c for c in feature_names if c not in last.columns]
        if missing:
            raise ValueError(f"Missing engineered RF feature columns: {missing}")
        X_last = last[feature_names].astype(float)
        nan_cols = X_last.columns[X_last.isna().any()].tolist()
        if nan_cols:
            raise ValueError(f"RF last-row features contain NaN: {nan_cols}")
        ret_pred = float(model.predict(X_last.to_numpy())[0])
        last_close = float(df["close_btc"].iloc[-1])
        predicted_close = last_close * math.exp(ret_pred)
        return float(predicted_close)

    def lstm_model(self, force_refresh=False):
        return self.naive(force_refresh=force_refresh)

    def tft_model(self, force_refresh=False):
        return self.naive(force_refresh=force_refresh)