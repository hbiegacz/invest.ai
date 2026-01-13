from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

CRYPTO_ASSETS = ["btc", "eth", "bnb", "xrp"]
INDEX_ASSETS = ["spx"]
ALL_ASSETS = CRYPTO_ASSETS + INDEX_ASSETS
EWM_SPAN = 7
FEATURE_COLUMNS: tuple[str, ...] = (
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
)


@dataclass(frozen=True)
class LSTMConfig:
    lookback: int
    target: str
    close_col: str
    feature_cols: tuple[str, ...]
    hidden_size: int
    num_layers: int
    dropout: float
    batch_size: int
    lr: float
    epochs: int
    weight_decay: float
    grad_clip: float
    val_ratio: float
    patience: int
    seed: int
    device: str
    target_scaling: bool


@dataclass(frozen=True)
class StandardScalerState:
    mean_: np.ndarray
    std_: np.ndarray
    feature_cols: tuple[str, ...]

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.std_

    @staticmethod
    def fit(df: pd.DataFrame, feature_cols: tuple[str, ...]) -> "StandardScalerState":
        X = df.loc[:, feature_cols].astype(float).to_numpy()
        mean_ = X.mean(axis=0)
        std_ = X.std(axis=0)
        std_ = np.where(std_ < 1e-12, 1.0, std_)
        return StandardScalerState(
            mean_=mean_, std_=std_, feature_cols=tuple(feature_cols)
        )


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _log_return(series: pd.Series) -> pd.Series:
    return np.log(series).diff()


def _dlog1p(series: pd.Series) -> pd.Series:
    return np.log1p(series).diff()


def _ewm_mean(series: pd.Series, span: int, shift: int = 1) -> pd.Series:
    s = series.shift(shift)
    return s.ewm(span=span, adjust=False).mean()


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for a in ALL_ASSETS:
        hi, lo = f"high_{a}", f"low_{a}"
        if hi in out.columns and lo in out.columns:
            out[f"hl2_{a}"] = (out[lo] + out[hi]) / 2.0
    volume_cols = [f"volume_{a}" for a in ALL_ASSETS if f"volume_{a}" in out.columns]
    if volume_cols:
        out["volume_sum"] = out[volume_cols].sum(axis=1)
    trades_cols = [
        f"num_trades_{a}" for a in CRYPTO_ASSETS if f"num_trades_{a}" in out.columns
    ]
    if trades_cols:
        out["num_trades_sum"] = out[trades_cols].sum(axis=1)
    for a in ALL_ASSETS:
        c = f"close_{a}"
        if c in out.columns:
            out[f"ret_close_{a}"] = _log_return(out[c])
        h = f"hl2_{a}"
        if h in out.columns:
            out[f"ret_hl2_{a}"] = _log_return(out[h])
    if "volume_sum" in out.columns:
        out["dlog_volume_sum"] = _dlog1p(out["volume_sum"])
    if "num_trades_sum" in out.columns:
        out["dlog_num_trades_sum"] = _dlog1p(out["num_trades_sum"])
    span = EWM_SPAN
    for a in ALL_ASSETS:
        rc = f"ret_close_{a}"
        if rc in out.columns:
            out[f"ewm_ret_close_{a}_s{span}"] = _ewm_mean(out[rc], span=span, shift=1)
        rh = f"ret_hl2_{a}"
        if rh in out.columns:
            out[f"ewm_ret_hl2_{a}_s{span}"] = _ewm_mean(out[rh], span=span, shift=1)
    if "dlog_volume_sum" in out.columns:
        out[f"ewm_dlog_volume_sum_s{span}"] = _ewm_mean(
            out["dlog_volume_sum"], span=span, shift=1
        )
    if "dlog_num_trades_sum" in out.columns:
        out[f"ewm_dlog_num_trades_sum_s{span}"] = _ewm_mean(
            out["dlog_num_trades_sum"], span=span, shift=1
        )
    if "gdp" in out.columns:
        gdp_lag1 = out["gdp"].shift(1)
        out["gdp_growth"] = gdp_lag1.pct_change()
    if "unrate" in out.columns:
        un_lag1 = out["unrate"].shift(1)
        out["unrate_change"] = un_lag1 - un_lag1.shift(1)
    return out


def build_target_ret_btc_next(df: pd.DataFrame, close_col: str) -> pd.DataFrame:
    out = df.copy()
    ret_btc = np.log(out[close_col].astype(float)).diff()
    out["ret_btc_next"] = ret_btc.shift(-1)
    return out


def make_sequences(
    df: pd.DataFrame,
    feature_cols: tuple[str, ...],
    target_col: str,
    lookback: int,
) -> tuple[np.ndarray, np.ndarray]:
    if lookback < 2:
        raise ValueError("lookback must be >= 2")
    X_all = df.loc[:, feature_cols].astype(float).to_numpy()
    y_all = df.loc[:, target_col].astype(float).to_numpy()
    N = len(df) - lookback + 1
    if N <= 0:
        raise ValueError("Not enough rows to build any sequence for given lookback.")
    X_seq = np.zeros((N, lookback, len(feature_cols)), dtype=np.float32)
    y_seq = np.zeros((N,), dtype=np.float32)
    for i in range(N):
        j = i + lookback - 1
        X_seq[i] = X_all[i : i + lookback]
        y_seq[i] = y_all[j]
    return X_seq, y_seq


class SequenceDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.head(last).squeeze(-1)


def train_lstm(df_raw: pd.DataFrame, cfg: LSTMConfig) -> dict:
    if not cfg.feature_cols:
        raise ValueError("cfg.feature_cols is empty")
    set_seed(cfg.seed)
    df = df_raw.sort_values("open_time").reset_index(drop=True)
    df = build_features(df)
    df = build_target_ret_btc_next(df, close_col=cfg.close_col)
    required = list(cfg.feature_cols) + [cfg.target]
    df = df.dropna(subset=required).reset_index(drop=True)
    split_idx = int(len(df) * (1.0 - cfg.val_ratio))
    df_train = df.iloc[:split_idx].copy()
    df_val = df.iloc[split_idx:].copy()
    scaler = StandardScalerState.fit(df_train, cfg.feature_cols)

    def apply_scaler(d: pd.DataFrame) -> pd.DataFrame:
        out = d.copy()
        cols = list(cfg.feature_cols)
        X = out[cols].to_numpy(dtype=np.float64)
        out[cols] = scaler.transform(X).astype(np.float32)
        return out

    df_train = apply_scaler(df_train)
    df_val = apply_scaler(df_val)
    X_tr, y_tr = make_sequences(df_train, cfg.feature_cols, cfg.target, cfg.lookback)
    X_va, y_va = make_sequences(df_val, cfg.feature_cols, cfg.target, cfg.lookback)
    baseline_val_mae = float(np.mean(np.abs(y_va)))
    y_mean = float(np.mean(y_tr)) if cfg.target_scaling else 0.0
    y_std = float(np.std(y_tr)) if cfg.target_scaling else 1.0
    if y_std < 1e-12:
        y_std = 1.0
    if cfg.target_scaling:
        y_tr = ((y_tr - y_mean) / y_std).astype(np.float32)
        y_va = ((y_va - y_mean) / y_std).astype(np.float32)
    train_loader = DataLoader(
        SequenceDataset(X_tr, y_tr),
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0,
    )
    val_loader = DataLoader(
        SequenceDataset(X_va, y_va),
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0,
    )
    device = torch.device(cfg.device)
    model = LSTMRegressor(
        n_features=len(cfg.feature_cols),
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
    ).to(device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=5
    )
    loss_fn = nn.L1Loss()
    best_val_mae = float("inf")
    best_state = None
    no_improve = 0
    mae_scale = float(y_std) if cfg.target_scaling else 1.0
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
            tr_abs_sum += torch.abs(pred - yb).sum().item() * mae_scale
            tr_count += yb.numel()
        model.eval()
        va_abs_sum = 0.0
        va_count = 0
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb = Xb.to(device)
                yb = yb.to(device)
                pred = model(Xb)
                va_abs_sum += torch.abs(pred - yb).sum().item() * mae_scale
                va_count += yb.numel()
        tr_mae = tr_abs_sum / max(1, tr_count)
        va_mae = va_abs_sum / max(1, va_count)
        scheduler.step(va_mae)
        if va_mae < best_val_mae:
            best_val_mae = va_mae
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            no_improve = 0
        else:
            no_improve += 1
        print(
            f"[epoch {epoch+1:02d}/{cfg.epochs}] "
            f"train_mae={tr_mae:.6f} "
            f"val_mae={va_mae:.6f} "
            f"baseline_val_mae={baseline_val_mae:.6f} "
            f"best_val_mae={best_val_mae:.6f}"
        )
        if no_improve >= cfg.patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    if best_state is None:
        raise RuntimeError("Training failed: best_state is None")
    return {
        "cfg": asdict(cfg),
        "scaler": {
            "mean_": scaler.mean_.tolist(),
            "std_": scaler.std_.tolist(),
            "feature_cols": list(scaler.feature_cols),
        },
        "target_scaler": {
            "enabled": bool(cfg.target_scaling),
            "mean_": float(y_mean),
            "std_": float(y_std),
        },
        "model_state": best_state,
        "best_val_mae": float(best_val_mae),
        "baseline_val_mae": float(baseline_val_mae),
    }


def predict_next_close(df_raw: pd.DataFrame, artifact: dict) -> float:
    cfg = LSTMConfig(**artifact["cfg"])
    scaler_d = artifact["scaler"]
    scaler = StandardScalerState(
        mean_=np.array(scaler_d["mean_"], dtype=np.float64),
        std_=np.array(scaler_d["std_"], dtype=np.float64),
        feature_cols=tuple(scaler_d["feature_cols"]),
    )
    ts = artifact.get("target_scaler", {"enabled": False, "mean_": 0.0, "std_": 1.0})
    y_mean = float(ts.get("mean_", 0.0))
    y_std = float(ts.get("std_", 1.0))
    if y_std < 1e-12:
        y_std = 1.0
    target_scaling_enabled = bool(ts.get("enabled", False))
    df = df_raw.sort_values("open_time").reset_index(drop=True)
    df = build_features(df)
    df = build_target_ret_btc_next(df, close_col=cfg.close_col)
    df = df.dropna(subset=list(cfg.feature_cols)).reset_index(drop=True)
    if len(df) < cfg.lookback:
        raise ValueError(
            f"Not enough rows for inference lookback window. Have {len(df)}, need {cfg.lookback}"
        )
    df_last = df.iloc[-cfg.lookback :].copy()
    X = df_last.loc[:, cfg.feature_cols].astype(float).to_numpy()
    X = (
        scaler.transform(X)
        .astype(np.float32)
        .reshape(1, cfg.lookback, len(cfg.feature_cols))
    )
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
        pred_scaled = float(model(torch.from_numpy(X).to(device)).cpu().item())
    pred_ret = pred_scaled * y_std + y_mean if target_scaling_enabled else pred_scaled
    last_close = float(df[cfg.close_col].iloc[-1])
    predicted_close = last_close * float(np.exp(pred_ret))
    return float(predicted_close)


def save_artifact(artifact: dict, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, path)


if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parents[1]
    parquet_path = ROOT / "backend" / "data" / "historical_data.parquet"
    out_path = ROOT / "backend" / "data" / "lstm_btc.pt"
    df = (
        pd.read_parquet(parquet_path, engine="pyarrow")
        .sort_values("open_time")
        .reset_index(drop=True)
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # best_val_mae=0.015663
    cfg = LSTMConfig(
        lookback=96,
        target="ret_btc_next",
        close_col="close_btc",
        feature_cols=FEATURE_COLUMNS,
        hidden_size=64,
        num_layers=2,
        dropout=0.15,
        batch_size=64,
        lr=8e-4,
        epochs=100,
        weight_decay=5e-4,
        grad_clip=1.0,
        val_ratio=0.2,
        patience=15,
        seed=314,
        device=device,
        target_scaling=True,
    )
    print(f"device={device}")
    print(f"lookback={cfg.lookback} seed={cfg.seed}")
    print(f"hidden={cfg.hidden_size} layers={cfg.num_layers} dropout={cfg.dropout}")
    print(f"lr={cfg.lr} weight_decay={cfg.weight_decay} batch_size={cfg.batch_size}")
    print(f"features={len(cfg.feature_cols)} (drop_ret_btc)")
    artifact = train_lstm(df, cfg)
    artifact["cfg"]["device"] = device
    pred = predict_next_close(df, artifact)
    print("\n==============================")
    print("RESULT")
    print("==============================")
    print(f"Feature count: {len(cfg.feature_cols)}")
    print(f"Baseline val MAE (pred=0): {artifact['baseline_val_mae']:.6f}")
    print(f"Best val MAE: {artifact['best_val_mae']:.6f}")
    print(f"Predicted next close_btc: {pred}")
    save_artifact(artifact, out_path)
    print(f"\nSaved artifact to: {out_path}")
