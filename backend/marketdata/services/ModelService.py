import os
import math
import re
import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import load
from django.conf import settings

from .DataReaderService import DataReaderService


class ModelService:
    def __init__(self):
        self.data_reader = DataReaderService()
        self._lstm_artifact_cache = None
        self._tft_artifact_cache = None

    def _ensure_fresh_file_if_needed(self, force_refresh: bool = False) -> None:
        if force_refresh or not os.path.exists(self.data_reader.file_path):
            self.data_reader._refresh_data_file()

    def _read_parquet(self, columns: list[str]) -> pd.DataFrame:
        """
        Reads parquet with a refresh retry if file is missing.
        """
        try:
            return pd.read_parquet(
                self.data_reader.file_path, columns=columns, engine="pyarrow"
            )
        except FileNotFoundError:
            self.data_reader._refresh_data_file()
            return pd.read_parquet(
                self.data_reader.file_path, columns=columns, engine="pyarrow"
            )

    def _require_columns(self, df: pd.DataFrame, cols: list[str]) -> None:
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in parquet: {missing}")

    def naive(self, force_refresh: bool = False) -> float:
        self._ensure_fresh_file_if_needed(force_refresh=force_refresh)
        df = self._read_parquet(columns=["close_btc"])
        if df.empty:
            raise ValueError(
                "No data available in historical_data.parquet for naive BTC model."
            )
        if "close_btc" not in df.columns:
            raise ValueError("Column 'close_btc' not found in historical_data.parquet.")
        return float(df["close_btc"].iloc[-1])

    def linear_regression(self, force_refresh: bool = False) -> float:
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
            raise ValueError(
                "Pipeline does not contain a 'scaler' step named 'scaler'."
            )

        model_features = list(getattr(scaler, "feature_names_in_", payload["features"]))
        cols_to_read = list(dict.fromkeys(model_features + ["open_time", "close_btc"]))

        df = self._read_parquet(columns=cols_to_read)
        if df.empty:
            raise ValueError(
                "No data available in historical_data.parquet for linear regression model."
            )

        missing = [c for c in model_features if c not in df.columns]
        if missing:
            raise ValueError(f"Missing feature columns in parquet: {missing}")

        df = df.sort_values("open_time").reset_index(drop=True)
        last_row_df = df.iloc[[-1]]

        X_last = last_row_df[model_features].astype(float).to_numpy()
        last_close = float(last_row_df["close_btc"].iloc[0])

        ret_pred = float(pipeline.predict(X_last)[0])
        predicted_close = last_close * (1.0 + ret_pred)
        return float(predicted_close)

    def _rf_base_columns(self) -> list[str]:
        assets_all = ["btc", "eth", "bnb", "xrp", "spx"]
        assets_crypto = ["btc", "eth", "bnb", "xrp"]
        cols = ["open_time", "gdp", "unrate"]
        for a in assets_all:
            cols += [f"high_{a}", f"low_{a}", f"close_{a}", f"volume_{a}"]
        for a in assets_crypto:
            cols += [f"num_trades_{a}"]
        return list(dict.fromkeys(cols))

    def _parse_rf_spans_windows(
        self, feature_names: list[str]
    ) -> tuple[list[int], list[int]]:
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

    def _compute_common_features(
        self, df: pd.DataFrame, spans: list[int], vol_windows: list[int]
    ) -> pd.DataFrame:
        """
        Builds the same style of engineered features as random_forest_train.load_dataset().
        It is intentionally broad, so it can serve RF + TFT.
        """
        df = df.sort_values("open_time").reset_index(drop=True)

        assets_all = ["btc", "eth", "bnb", "xrp", "spx"]
        assets_crypto = ["btc", "eth", "bnb", "xrp"]

        for a in assets_all:
            df[f"hl2_{a}"] = (df[f"low_{a}"] + df[f"high_{a}"]) / 2.0

        df["volume_sum"] = df[[f"volume_{a}" for a in assets_all]].sum(axis=1)
        df["num_trades_sum"] = df[[f"num_trades_{a}" for a in assets_crypto]].sum(
            axis=1
        )

        for a in assets_all:
            df[f"ret_close_{a}"] = np.log(df[f"close_{a}"].astype(float)).diff()
            df[f"ret_hl2_{a}"] = np.log(df[f"hl2_{a}"].astype(float)).diff()

        df["dlog_volume_sum"] = np.log1p(df["volume_sum"].astype(float)).diff()
        df["dlog_num_trades_sum"] = np.log1p(df["num_trades_sum"].astype(float)).diff()

        for span in spans:
            for a in assets_all:
                df[f"ewm_ret_close_{a}_s{span}"] = (
                    df[f"ret_close_{a}"].shift(1).ewm(span=span, adjust=False).mean()
                )
                df[f"ewm_ret_hl2_{a}_s{span}"] = (
                    df[f"ret_hl2_{a}"].shift(1).ewm(span=span, adjust=False).mean()
                )

            df[f"ewm_dlog_volume_sum_s{span}"] = (
                df["dlog_volume_sum"].shift(1).ewm(span=span, adjust=False).mean()
            )
            df[f"ewm_dlog_num_trades_sum_s{span}"] = (
                df["dlog_num_trades_sum"].shift(1).ewm(span=span, adjust=False).mean()
            )

        for w in vol_windows:
            df[f"roll_std_ret_close_btc_w{w}"] = (
                df["ret_close_btc"].shift(1).rolling(window=w, min_periods=w).std()
            )

        df["gdp_lag1"] = df["gdp"].shift(1)
        df["unrate_lag1"] = df["unrate"].shift(1)
        df["gdp_growth"] = df["gdp_lag1"].pct_change()
        df["unrate_change"] = df["unrate_lag1"] - df["unrate_lag1"].shift(1)

        df["ret_btc"] = np.log(df["close_btc"].astype(float)).diff()
        df["ret_btc_next"] = df["ret_btc"].shift(-1)

        return df

    def random_forest(self, force_refresh: bool = False) -> float:
        self._ensure_fresh_file_if_needed(force_refresh=force_refresh)

        model_path = Path(settings.BASE_DIR) / "data" / "random_forest_btc.pkl"
        if not model_path.exists():
            raise FileNotFoundError(f"Trained RF model file not found at: {model_path}")

        payload = load(model_path)
        model = payload.get("model")
        feature_names = payload.get("features")
        if model is None or not feature_names:
            raise ValueError("RF payload missing 'model' or 'features'.")

        spans, vol_windows = self._parse_rf_spans_windows(list(feature_names))

        df = self._read_parquet(columns=self._rf_base_columns())
        if df.empty:
            raise ValueError(
                "No data available in historical_data.parquet for random forest model."
            )

        df_feat = self._compute_common_features(
            df, spans=spans, vol_windows=vol_windows
        )
        last = df_feat.iloc[[-1]]

        missing = [c for c in feature_names if c not in last.columns]
        if missing:
            raise ValueError(f"Missing engineered RF feature columns: {missing}")

        X_last = last[feature_names].astype(float)
        nan_cols = X_last.columns[X_last.isna().any()].tolist()
        if nan_cols:
            raise ValueError(f"RF last-row features contain NaN: {nan_cols}")

        ret_pred = float(model.predict(X_last.to_numpy())[0])
        last_close = float(df_feat["close_btc"].iloc[-1])
        predicted_close = last_close * math.exp(ret_pred)
        return float(predicted_close)

    def _load_lstm_artifact(self) -> dict:
        if self._lstm_artifact_cache is not None:
            return self._lstm_artifact_cache
        try:
            import torch
        except Exception as e:
            raise RuntimeError(
                f"torch is required for LSTM inference but could not be imported: {e}"
            )
        model_path = Path(settings.BASE_DIR) / "data" / "lstm_btc.pt"
        if not model_path.exists():
            raise FileNotFoundError(f"LSTM artifact not found at: {model_path}")
        artifact = torch.load(model_path, map_location="cpu")
        self._lstm_artifact_cache = artifact
        return artifact

    def lstm_model(self, force_refresh: bool = False) -> float:
        self._ensure_fresh_file_if_needed(force_refresh=force_refresh)
        artifact = self._load_lstm_artifact()
        try:
            import torch
            import torch.nn as nn
        except Exception as e:
            raise RuntimeError(
                f"torch is required for LSTM inference but could not be imported: {e}"
            )
        cfg = artifact.get("cfg")
        scaler = artifact.get("scaler")
        state = artifact.get("model_state")
        target_scaler = artifact.get(
            "target_scaler", {"enabled": False, "mean_": 0.0, "std_": 1.0}
        )
        if not cfg or not scaler or not state:
            raise ValueError(
                "Invalid LSTM artifact: missing 'cfg', 'scaler' or 'model_state'."
            )
        lookback = int(cfg["lookback"])
        hidden_size = int(cfg["hidden_size"])
        num_layers = int(cfg["num_layers"])
        dropout = float(cfg["dropout"])
        feature_cols = list(scaler["feature_cols"])
        mean_ = np.array(scaler["mean_"], dtype=np.float64)
        std_ = np.array(scaler["std_"], dtype=np.float64)
        if len(feature_cols) != len(mean_) or len(feature_cols) != len(std_):
            raise ValueError(
                "Invalid LSTM scaler state: mean/std length mismatch with feature_cols."
            )
        df_raw = self._read_parquet(columns=self._rf_base_columns())
        if df_raw.empty:
            raise ValueError(
                "No data available in historical_data.parquet for LSTM model."
            )
        spans, _ = self._parse_rf_spans_windows(feature_cols)
        if not spans:
            spans = [7]
        df_feat = self._compute_common_features(df_raw, spans=spans, vol_windows=[])
        df_feat = df_feat.sort_values("open_time").reset_index(drop=True)
        df_feat = df_feat.dropna(subset=feature_cols).reset_index(drop=True)
        if len(df_feat) < lookback:
            raise ValueError(
                f"Not enough rows for LSTM inference. need>={lookback}, have={len(df_feat)}"
            )
        df_last = df_feat.iloc[-lookback:].copy()
        X = df_last.loc[:, feature_cols].astype(float).to_numpy(dtype=np.float64)
        X = (X - mean_) / std_
        if not np.isfinite(X).all():
            raise ValueError(
                "LSTM features contain NaN/inf after scaling (check data + FE)."
            )
        X = X.astype(np.float32).reshape(1, lookback, len(feature_cols))

        class LSTMRegressor(nn.Module):
            def __init__(
                self, n_features: int, hidden_size: int, num_layers: int, dropout: float
            ):
                super().__init__()
                self.lstm = nn.LSTM(
                    input_size=n_features,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout=dropout if num_layers > 1 else 0.0,
                    batch_first=True,
                )
                self.head = nn.Sequential(
                    nn.Linear(hidden_size, hidden_size // 2),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_size // 2, 1),
                )

            def forward(self, x):
                out, _ = self.lstm(x)
                last = out[:, -1, :]
                return self.head(last).squeeze(-1)

        model = LSTMRegressor(
            n_features=len(feature_cols),
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )
        model.load_state_dict(state)
        model.eval()
        with torch.no_grad():
            pred_scaled = float(model(torch.from_numpy(X)).cpu().item())
        if bool(target_scaler.get("enabled", False)):
            y_mean = float(target_scaler.get("mean_", 0.0))
            y_std = float(target_scaler.get("std_", 1.0)) or 1.0
            pred_ret = pred_scaled * y_std + y_mean
        else:
            pred_ret = pred_scaled
        last_close = float(df_feat["close_btc"].iloc[-1])
        try:
            predicted_close = last_close * math.exp(pred_ret)
        except OverflowError:
            raise ValueError(f"Overflow in exp(pred_ret). pred_ret={pred_ret}")
        return float(predicted_close)

    def _load_tft_artifact(self) -> dict:
        if self._tft_artifact_cache is not None:
            return self._tft_artifact_cache

        try:
            from darts.models import TFTModel
        except Exception as e:
            raise RuntimeError(
                f"darts is required for TFT inference but could not be imported: {e}"
            )

        tft_dir = Path(settings.BASE_DIR) / "data" / "tft_btc"
        if not tft_dir.exists():
            raise FileNotFoundError(f"TFT artifact dir not found at: {tft_dir}")

        meta_path = tft_dir / "metadata.json"
        scalers_path = tft_dir / "scalers.joblib"
        model_path = tft_dir / "tft_model"

        if (
            not meta_path.exists()
            or not scalers_path.exists()
            or not model_path.exists()
        ):
            raise FileNotFoundError(
                "TFT artifact is incomplete. Expected: metadata.json, scalers.joblib, tft_model"
            )

        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        scalers = load(scalers_path)

        model = TFTModel.load(str(model_path))

        art = {"dir": tft_dir, "metadata": metadata, "scalers": scalers, "model": model}
        self._tft_artifact_cache = art
        return art

    def _add_time_features(self, df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        if "open_time" not in df.columns:
            raise ValueError("Expected 'open_time' column in DataFrame.")

        out = df.copy()
        t = pd.to_datetime(out["open_time"], utc=False, errors="coerce")
        out = out.loc[~t.isna()].copy()
        t = pd.to_datetime(out["open_time"], utc=False, errors="coerce")

        dow = t.dt.dayofweek.astype(int).to_numpy()
        week = t.dt.isocalendar().week.astype(int).to_numpy()

        out["dow_sin"] = np.sin(2.0 * np.pi * dow / 7.0)
        out["dow_cos"] = np.cos(2.0 * np.pi * dow / 7.0)
        out["week_sin"] = np.sin(2.0 * np.pi * week / 52.0)
        out["week_cos"] = np.cos(2.0 * np.pi * week / 52.0)

        return out, ["dow_sin", "dow_cos", "week_sin", "week_cos"]

    def tft_model(self, force_refresh: bool = False) -> float:
        """
        Predicts next close_btc using the saved Darts TFT artifact.
        Returns predicted close price (not return).
        """
        self._ensure_fresh_file_if_needed(force_refresh=force_refresh)

        try:
            from darts import TimeSeries
        except Exception as e:
            raise RuntimeError(
                f"darts is required for TFT inference but could not be imported: {e}"
            )

        art = self._load_tft_artifact()
        model = art["model"]
        metadata = art["metadata"]
        scaler_target = art["scalers"]["scaler_target"]
        scaler_cov = art["scalers"]["scaler_cov"]

        features_base = list(metadata.get("features_base", []))
        covariates = list(metadata.get("covariates", []))
        target_col = str(metadata.get("target", "ret_btc_next"))

        if not features_base:
            raise ValueError("TFT metadata missing 'features_base'.")
        if not covariates:
            raise ValueError("TFT metadata missing 'covariates'.")

        spans, vol_windows = self._parse_rf_spans_windows(features_base)

        df_raw = self._read_parquet(columns=self._rf_base_columns())
        if df_raw.empty:
            raise ValueError(
                "No data available in historical_data.parquet for TFT model."
            )

        df_feat = self._compute_common_features(
            df_raw, spans=spans, vol_windows=vol_windows
        )

        df_feat["open_time"] = pd.to_datetime(
            df_feat["open_time"], utc=False, errors="coerce"
        )
        df_feat = df_feat.dropna(subset=["open_time"]).sort_values("open_time")
        df_feat = df_feat.drop_duplicates(
            subset=["open_time"], keep="last"
        ).reset_index(drop=True)

        df_feat, _time_cols = self._add_time_features(df_feat)

        need_cols = list(
            dict.fromkeys(
                ["open_time", "close_btc"] + features_base + [target_col] + covariates
            )
        )
        self._require_columns(df_feat, need_cols)

        df_feat = df_feat.dropna(
            subset=list(dict.fromkeys(features_base + [target_col] + covariates))
        ).reset_index(drop=True)
        if df_feat.empty:
            raise ValueError(
                "TFT preprocessing produced empty DataFrame (likely too many NaNs after feature engineering)."
            )

        df_feat["_t"] = np.arange(len(df_feat), dtype=np.int64)

        target = TimeSeries.from_dataframe(
            df_feat, time_col="_t", value_cols=[target_col]
        )
        past_cov = TimeSeries.from_dataframe(
            df_feat, time_col="_t", value_cols=covariates
        )

        target_s = scaler_target.transform(target)
        cov_s = scaler_cov.transform(past_cov)

        pred_s = model.predict(
            n=1, series=target_s, past_covariates=cov_s, verbose=False
        )
        pred = scaler_target.inverse_transform(pred_s)

        pred_ret = float(pred.values(copy=False).reshape(-1)[0])

        last_close = float(df_feat["close_btc"].iloc[-1])
        predicted_close = last_close * math.exp(pred_ret)
        return float(predicted_close)
