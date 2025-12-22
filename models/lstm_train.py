"""
Co pozmieniałam:
1. Dodałam możliwość utworzenia więcej features, które są w Random Forest (log-returny, wolumeny, makro itp.). 
2. Dodałam warstwę Dropout na końcu, czyli 'wyłączanie' części neuronów w trakcie nauki, aby unikać przeuczenia. 
3. Early Stopping - po 15 epokach bez poprawy zatrzymuje trening.
4. Scheduler - zwalnia naukę (zmniejsza LR), jeśli MAE się nie poprawiło przez ostatnie 5 epok.
5. Troszkę inne parametry

I teraz MAE jest na 0.016093.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

CRYPTO_ASSETS = ["btc", "eth", "bnb", "xrp"]
INDEX_ASSETS = ["spx"]
ALL_ASSETS = CRYPTO_ASSETS + INDEX_ASSETS

EWM_SPANS = [7]
VOL_WINDOWS = [21]

MANUAL_FEATURE_COLUMNS = [
    "ret_close_bnb",
    "ret_close_btc", 
    "ret_close_eth", 
    "ret_close_spx", 
    "ret_close_xrp",
    "ret_hl2_bnb", 
    "ret_hl2_btc", 
    "ret_hl2_eth", 
    "ret_hl2_spx", 
    "ret_hl2_xrp",
    "ewm_dlog_num_trades_sum_s7", 
    "ewm_dlog_volume_sum_s7",
    "ewm_ret_close_bnb_s7", 
    "ewm_ret_close_btc_s7", 
    "ewm_ret_close_eth_s7", 
    "ewm_ret_close_spx_s7", 
    "ewm_ret_close_xrp_s7",
    "ewm_ret_hl2_bnb_s7", 
    "ewm_ret_hl2_btc_s7", 
    "ewm_ret_hl2_eth_s7", 
    "ewm_ret_hl2_spx_s7", 
    "ewm_ret_hl2_xrp_s7",
    "dlog_num_trades_sum", 
    "dlog_volume_sum",
    "gdp_growth", 
    "unrate_change", 
    "ret_btc"
]

@dataclass(frozen=True)
class LSTMConfig:
    lookback: int = 60
    target: str = "ret_btc_next"
    close_col: str = "close_btc"
    feature_cols: tuple[str, ...] = tuple()
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    batch_size: int = 64
    lr: float = 1e-3
    epochs: int = 100     
    weight_decay: float = 1e-3
    grad_clip: float = 1.0
    val_ratio: float = 0.2
    patience: int = 15   
    seed: int = 42
    device: str = "cpu"

@dataclass
class StandardScalerState:
    mean_: np.ndarray
    std_: np.ndarray
    feature_cols: tuple[str, ...]

    def transform(self, X):
        return (X - self.mean_) / self.std_

    @staticmethod
    def fit(df, feature_cols):
        X = df.loc[:, feature_cols].astype(float).to_numpy()
        mean_ = X.mean(axis=0)
        std_ = X.std(axis=0)
        std_ = np.where(std_ < 1e-12, 1.0, std_)
        return StandardScalerState(mean_=mean_, std_=std_, feature_cols=tuple(feature_cols))


def _log_return(series: pd.Series) -> pd.Series:
    """
    Computes log-return: log(x_t) - log(x_{t-1}).
    Assumes series values are > 0.
    """
    return np.log(series).diff()


def _dlog1p(series: pd.Series) -> pd.Series:
    """
    Computes change in log(1 + x): log1p(x_t) - log1p(x_{t-1}).
    Works with zeros and stabilizes heavy-tailed series like volume / trades.
    """
    return np.log1p(series).diff()


def _ewm_mean(series: pd.Series, span: int, shift: int = 1) -> pd.Series:
    """
    Exponentially weighted moving average computed from past values only.
    
    shift=1 -> at time t uses data up to t-1 (safe for "predict tomorrow" setups).
    """
    s = series.shift(shift)
    return s.ewm(span=span, adjust=False).mean()


def _rolling_std(series: pd.Series, window: int, shift: int = 1) -> pd.Series:
    """
    Rolling std computed from past values only (shift=1).
    window=7/21 captures volatility regime (short/medium horizon).
    """
    s = series.shift(shift)
    return s.rolling(window=window, min_periods=window).std()


def build_features(df, cfg):
    """
    Builds comprehensive features matching Random Forest implementation.
    Creates log returns, EWM features, rolling volatility, and macro features.
    """
    out = df.copy()
    
    # 1. Create HL2 (high-low average) for all assets
    for a in ALL_ASSETS:
        if f"high_{a}" in out.columns and f"low_{a}" in out.columns:
            out[f"hl2_{a}"] = (out[f"low_{a}"] + out[f"high_{a}"]) / 2.0
    
    # 2. Aggregate volume and trades
    volume_cols = [f"volume_{a}" for a in ALL_ASSETS if f"volume_{a}" in out.columns]
    if volume_cols:
        out["volume_sum"] = out[volume_cols].sum(axis=1)
    
    trades_cols = [f"num_trades_{a}" for a in CRYPTO_ASSETS if f"num_trades_{a}" in out.columns]
    if trades_cols:
        out["num_trades_sum"] = out[trades_cols].sum(axis=1)
    
    # 3. Compute log returns for close and hl2 prices
    for a in ALL_ASSETS:
        if f"close_{a}" in out.columns:
            out[f"ret_close_{a}"] = _log_return(out[f"close_{a}"])
        if f"hl2_{a}" in out.columns:
            out[f"ret_hl2_{a}"] = _log_return(out[f"hl2_{a}"])
    
    # 4. Compute dlog transformations for volume and trades
    if "volume_sum" in out.columns:
        out["dlog_volume_sum"] = _dlog1p(out["volume_sum"])
    if "num_trades_sum" in out.columns:
        out["dlog_num_trades_sum"] = _dlog1p(out["num_trades_sum"])
    
    # 5. Generate EWM features
    for span in EWM_SPANS:
        for a in ALL_ASSETS:
            if f"ret_close_{a}" in out.columns:
                out[f"ewm_ret_close_{a}_s{span}"] = _ewm_mean(out[f"ret_close_{a}"], span=span, shift=1)
            if f"ret_hl2_{a}" in out.columns:
                out[f"ewm_ret_hl2_{a}_s{span}"] = _ewm_mean(out[f"ret_hl2_{a}"], span=span, shift=1)
        
        if "dlog_volume_sum" in out.columns:
            out[f"ewm_dlog_volume_sum_s{span}"] = _ewm_mean(out["dlog_volume_sum"], span=span, shift=1)
        if "dlog_num_trades_sum" in out.columns:
            out[f"ewm_dlog_num_trades_sum_s{span}"] = _ewm_mean(out["dlog_num_trades_sum"], span=span, shift=1)
    
    for w in VOL_WINDOWS:
        if "ret_close_btc" in out.columns:
            out[f"roll_std_ret_close_btc_w{w}"] = _rolling_std(out["ret_close_btc"], window=w, shift=1)
    
    if "gdp" in out.columns:
        out["gdp_lag1"] = out["gdp"].shift(1)
        out["gdp_growth"] = out["gdp_lag1"].pct_change()
    
    if "unrate" in out.columns:
        out["unrate_lag1"] = out["unrate"].shift(1)
        out["unrate_change"] = out["unrate_lag1"] - out["unrate_lag1"].shift(1)
    
    return out

def build_target_ret_btc_next(df, close_col):
    out = df.copy()
    out["ret_btc"] = np.log(out[close_col].astype(float)).diff()
    out["ret_btc_next"] = out["ret_btc"].shift(-1)
    return out

def make_sequences(
    df,
    feature_cols,
    target_col,
    lookback,
):
    if lookback < 2:
        raise ValueError("lookback must be >= 2")
    df = df.dropna(subset=list(feature_cols) + [target_col]).reset_index(drop=True)
    X_all = df.loc[:, feature_cols].astype(float).to_numpy()
    y_all = df.loc[:, target_col].astype(float).to_numpy()
    ts_all = pd.to_datetime(df["open_time"]).astype("int64").to_numpy() // 1_000_000
    N = len(df) - lookback + 1
    if N <= 0:
        raise ValueError("Not enough rows to build any sequence for given lookback.")
    X_seq = np.zeros((N, lookback, len(feature_cols)), dtype=np.float32)
    y_seq = np.zeros((N,), dtype=np.float32)
    ts_seq = np.zeros((N,), dtype=np.int64)
    for i in range(N):
        j = i + lookback - 1
        X_seq[i] = X_all[i : i + lookback]
        y_seq[i] = y_all[j]
        ts_seq[i] = ts_all[j]
    return X_seq, y_seq, ts_seq

class SequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return int(self.X.shape[0])

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class LSTMRegressor(nn.Module):
    def __init__(self, n_features, hidden_size, num_layers, dropout):
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
        pred = self.head(last).squeeze(-1)
        return pred

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def train_lstm(df_raw, cfg):
    if not cfg.feature_cols:
        raise ValueError("cfg.feature_cols is empty. Wait for feature selection and then fill it.")
    set_seed(cfg.seed)
    df = df_raw.sort_values("open_time").reset_index(drop=True)
    print(f"Initial dataset size: {len(df)} rows")
    df = build_features(df, cfg)
    df = build_target_ret_btc_next(df, close_col=cfg.close_col)
    required_cols = list(cfg.feature_cols) + [cfg.target]
    df = df.dropna(subset=required_cols).reset_index(drop=True)
    print(f"After feature engineering and NaN removal: {len(df)} rows")
    if len(df) < cfg.lookback + 10:
        raise ValueError(f"Not enough data after feature engineering. Have {len(df)} rows, need at least {cfg.lookback + 10}")
    split_idx = int(len(df) * (1.0 - cfg.val_ratio))
    df_train = df.iloc[:split_idx].copy()
    df_val = df.iloc[split_idx:].copy()
    print(f"Train set: {len(df_train)} rows, Val set: {len(df_val)} rows")
    scaler = StandardScalerState.fit(df_train, cfg.feature_cols)
    def _apply_scaler(d):
        out = d.copy()
        cols = list(cfg.feature_cols)
        out[cols] = out[cols].astype(np.float64)
        X = out[cols].to_numpy(dtype=np.float64)
        out[cols] = scaler.transform(X).astype(np.float32)
        return out
    df_train_s = _apply_scaler(df_train)
    df_val_s = _apply_scaler(df_val)
    X_tr, y_tr, _ = make_sequences(df_train_s, cfg.feature_cols, cfg.target, cfg.lookback)
    X_va, y_va, _ = make_sequences(df_val_s, cfg.feature_cols, cfg.target, cfg.lookback)
    train_loader = DataLoader(SequenceDataset(X_tr, y_tr), batch_size=cfg.batch_size, shuffle=False)
    val_loader = DataLoader(SequenceDataset(X_va, y_va), batch_size=cfg.batch_size, shuffle=False)
    device = torch.device(cfg.device)
    model = LSTMRegressor(
        n_features=len(cfg.feature_cols),
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=5)
    loss_fn = nn.MSELoss()
    best_val_mae = float("inf")
    best_state = None
    no_improve_epochs = 0
    for epoch in range(cfg.epochs):
      model.train()
      tr_abs_sum = 0.0
      tr_count = 0
      for Xb, yb in train_loader:
          Xb = Xb.to(device)
          yb = yb.to(device)
          opt.zero_grad(set_to_none=True)
          pred = model(Xb)
          loss = loss_fn(pred, yb)
          loss.backward()
          if cfg.grad_clip and cfg.grad_clip > 0:
              torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
          opt.step()
          tr_abs_sum += torch.abs(pred - yb).sum().item()
          tr_count += yb.numel()
      model.eval()
      va_abs_sum = 0.0
      va_count = 0
      with torch.no_grad():
          for Xb, yb in val_loader:
              Xb = Xb.to(device)
              yb = yb.to(device)
              pred = model(Xb)
              va_abs_sum += torch.abs(pred - yb).sum().item()
              va_count += yb.numel()
      tr_mae = tr_abs_sum / max(1, tr_count)
      va_mae = va_abs_sum / max(1, va_count)  
      scheduler.step(va_mae)
    
      if va_mae < best_val_mae:
          best_val_mae = va_mae
          best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
          no_improve_epochs = 0
      else:
          no_improve_epochs += 1
          
      print(f"[epoch {epoch+1:02d}/{cfg.epochs}] train_mae={tr_mae:.6f} val_mae={va_mae:.6f} best_val_mae={best_val_mae:.6f}")
      
      if no_improve_epochs >= cfg.patience:
          print(f"Early stopping at epoch {epoch+1}")
          break
          
    if best_state is None:
        raise RuntimeError("Training failed: best_state is None")
    artifact = {
        "cfg": asdict(cfg),
        "scaler": {
            "mean_": scaler.mean_.tolist(),
            "std_": scaler.std_.tolist(),
            "feature_cols": list(scaler.feature_cols),
        },
        "model_state": best_state,
        "best_val_mae": best_val_mae,
    }
    return artifact


def predict_next_close(
    df_raw,
    artifact,
):
    cfg_d = artifact["cfg"]
    cfg = LSTMConfig(**cfg_d)
    scaler_d = artifact["scaler"]
    scaler = StandardScalerState(
        mean_=np.array(scaler_d["mean_"], dtype=np.float64),
        std_=np.array(scaler_d["std_"], dtype=np.float64),
        feature_cols=tuple(scaler_d["feature_cols"]),
    )
    df = df_raw.sort_values("open_time").reset_index(drop=True)
    df = build_features(df, cfg)
    df = build_target_ret_btc_next(df, close_col=cfg.close_col)
    required_cols = list(cfg.feature_cols) + [cfg.target]
    df = df.dropna(subset=required_cols).reset_index(drop=True)
    if len(df) < cfg.lookback:
        raise ValueError(f"Not enough rows for inference lookback window. Have {len(df)}, need {cfg.lookback}")
    df_last = df.iloc[-cfg.lookback:].copy()
    X = df_last.loc[:, cfg.feature_cols].astype(float).to_numpy()
    X = scaler.transform(X).astype(np.float32)
    X = X.reshape(1, cfg.lookback, len(cfg.feature_cols))
    device = torch.device(cfg.device)
    model = LSTMRegressor(
        n_features=len(cfg.feature_cols),
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
    ).to(device)
    model.load_state_dict(artifact["model_state"])
    model.eval()
    with torch.no_grad():
        pred_ret = float(model(torch.from_numpy(X).to(device)).cpu().item())
    last_close = float(df[cfg.close_col].iloc[-1])
    predicted_close = last_close * math.exp(pred_ret)
    return float(predicted_close)

def save_artifact(artifact, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, path)

def load_artifact(path):
    return torch.load(Path(path), map_location="cpu")

if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parents[1] 
    parquet_path = ROOT / "backend" / "data" / "historical_data.parquet"
    df = pd.read_parquet(parquet_path, engine="pyarrow")
    df = df.sort_values("open_time").reset_index(drop=True)
    
    feature_cols = MANUAL_FEATURE_COLUMNS
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = LSTMConfig(
        feature_cols=tuple(feature_cols),
        device=device,
    )
    artifact = train_lstm(df, cfg)
    pred = predict_next_close(df, artifact)
    print(f"Feature count: {len(feature_cols)}")
    print(f"Best val MAE: {artifact['best_val_mae']}")
    print(f"Predicted next close_btc: {pred}")
    out_path = ROOT / "backend" / "data" / "lstm_btc.pt"
    save_artifact(artifact, out_path)
    print(f"Saved artifact to: {out_path}")
