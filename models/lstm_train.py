from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

@dataclass(frozen=True)
class LSTMConfig:
    lookback: int = 60
    target: str = "ret_btc_next"
    close_col: str = "close_btc"
    feature_cols: tuple[str, ...] = tuple()
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.1
    batch_size: int = 64
    lr: float = 1e-3
    epochs: int = 20
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    val_ratio: float = 0.2
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


def build_features(df, cfg):
    return df

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
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
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
    df = build_features(df, cfg)
    df = build_target_ret_btc_next(df, close_col=cfg.close_col)
    split_idx = int(len(df) * (1.0 - cfg.val_ratio))
    df_train = df.iloc[:split_idx].copy()
    df_val = df.iloc[split_idx:].copy()
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
    loss_fn = nn.MSELoss()
    best_val_mae = float("inf")
    best_state = None
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
      if va_mae < best_val_mae:
          best_val_mae = va_mae
          best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
      print(f"[epoch {epoch+1}/{cfg.epochs}] train_mae={tr_mae:.6f} val_mae={va_mae:.6f} best_val_mae={best_val_mae:.6f}")
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
    if len(df) < cfg.lookback:
        raise ValueError("Not enough rows for inference lookback window.")
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
    exclude = {"open_time", "ret_btc", "ret_btc_next"}
    feature_cols = [
        c for c in df.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
    ]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = LSTMConfig(
        feature_cols=tuple(feature_cols),
        lookback=60,
        epochs=20,
        batch_size=128,
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
