from __future__ import annotations

import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from joblib import dump, load
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

from darts import TimeSeries, concatenate
from darts.dataprocessing.transformers import Scaler
from darts.models import TFTModel

from random_forest_train import (
    load_dataset as load_rf_dataset,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
)


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT_DIR / "backend" / "data" / "tft_btc"


RUN_MODE = "single"  # "single" | "grid"
GRID_MODE = "full"  # "random" | "full"

OUTPUT_DIR = DEFAULT_OUTPUT_DIR

SEED = 42
TEST_SIZE = 0.2
VAL_RATIO = 0.15

EPOCHS = 30
WEIGHT_DECAY = 0.0

EVAL_STRIDE = 5
EVAL_MAX_POINTS = 250

EARLY_STOPPING_PATIENCE = 6
EARLY_STOPPING_MIN_DELTA = 0.0

RANDOM_TRIALS = 16

SINGLE_INPUT_CHUNK_LENGTH = 30
SINGLE_OUTPUT_CHUNK_LENGTH = 1
SINGLE_HIDDEN_SIZE = 16
SINGLE_LSTM_LAYERS = 2
SINGLE_NUM_ATTENTION_HEADS = 4
SINGLE_DROPOUT = 0.0
SINGLE_BATCH_SIZE = 64
SINGLE_LR = 1e-3


@dataclass(frozen=True)
class TFTConfig:
    input_chunk_length: int = 60
    output_chunk_length: int = 1

    hidden_size: int = 32
    lstm_layers: int = 1
    num_attention_heads: int = 4
    dropout: float = 0.1

    batch_size: int = 64
    epochs: int = 30
    lr: float = 1e-3
    weight_decay: float = 0.0

    test_size: float = 0.2
    val_ratio: float = 0.15
    seed: int = 42

    eval_stride: int = 1
    eval_max_points: int | None = None

    early_stopping_patience: int = 6
    early_stopping_min_delta: float = 0.0


@dataclass(frozen=True)
class SearchSpace:
    input_chunk_length: tuple[int, ...] = (30, 60, 90)
    hidden_size: tuple[int, ...] = (16, 96)
    lstm_layers: tuple[int, ...] = (2,)
    num_attention_heads: tuple[int, ...] = (1, 4)
    dropout: tuple[float, ...] = (0.0, 0.2)
    lr: tuple[float, ...] = (3e-4, 1e-3)
    batch_size: tuple[int, ...] = (64,)

    # input_chunk_length: tuple[int, ...] = (30,)
    # hidden_size: tuple[int, ...] = (16,)
    # lstm_layers: tuple[int, ...] = (2,)
    # num_attention_heads: tuple[int, ...] = (4,)
    # dropout: tuple[float, ...] = (0.0,)
    # lr: tuple[float, ...] = (1e-3,)
    # batch_size: tuple[int, ...] = (64,)


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def _make_early_stopping_callback(patience: int, min_delta: float):
    """
    Builds EarlyStopping callback if Lightning is available.
    Returns None if neither lightning.pytorch nor pytorch_lightning is installed.
    """
    try:
        from lightning.pytorch.callbacks import EarlyStopping

        return EarlyStopping(
            monitor="val_loss", patience=patience, min_delta=min_delta, mode="min"
        )
    except Exception:
        pass

    try:
        from pytorch_lightning.callbacks import EarlyStopping

        return EarlyStopping(
            monitor="val_loss", patience=patience, min_delta=min_delta, mode="min"
        )
    except Exception:
        return None


def _pick_pl_trainer_kwargs(cfg: TFTConfig) -> dict:
    """
    CPU-only trainer kwargs with optional early stopping.
    """
    try:
        n = os.cpu_count() or 1
        torch.set_num_threads(max(1, n - 1))
    except Exception:
        pass

    callbacks = []
    cb = _make_early_stopping_callback(
        cfg.early_stopping_patience, cfg.early_stopping_min_delta
    )
    if cb is not None:
        callbacks.append(cb)

    return {
        "accelerator": "cpu",
        "devices": 1,
        "enable_progress_bar": False,
        "logger": False,
        "enable_checkpointing": False,
        "callbacks": callbacks,
    }


def _add_time_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Adds cyclic calendar features based on open_time.
    """
    if "open_time" not in df.columns:
        raise ValueError("Expected 'open_time' column in DataFrame.")

    out = df.copy()
    t = pd.to_datetime(out["open_time"], utc=False, errors="coerce")
    if t.isna().any():
        out = out.loc[~t.isna()].copy()
        t = pd.to_datetime(out["open_time"], utc=False, errors="coerce")

    dow = t.dt.dayofweek.astype(int).to_numpy()
    week = t.dt.isocalendar().week.astype(int).to_numpy()

    out["dow_sin"] = np.sin(2.0 * np.pi * dow / 7.0)
    out["dow_cos"] = np.cos(2.0 * np.pi * dow / 7.0)
    out["week_sin"] = np.sin(2.0 * np.pi * week / 52.0)
    out["week_cos"] = np.cos(2.0 * np.pi * week / 52.0)

    cols = ["dow_sin", "dow_cos", "week_sin", "week_cos"]
    return out, cols


def build_timeseries(df: pd.DataFrame) -> tuple[TimeSeries, TimeSeries, list[str]]:
    """
    Builds target and past_covariates TimeSeries using a step index (0..N-1).
    """
    d = df.copy()
    d["open_time"] = pd.to_datetime(d["open_time"], utc=False, errors="coerce")
    d = d.dropna(subset=["open_time"]).sort_values("open_time")
    d = d.drop_duplicates(subset=["open_time"], keep="last").reset_index(drop=True)

    d, time_cols = _add_time_features(d)
    cov_cols = list(FEATURE_COLUMNS) + time_cols

    d["_t"] = np.arange(len(d), dtype=np.int64)

    target = TimeSeries.from_dataframe(d, time_col="_t", value_cols=[TARGET_COLUMN])
    past_cov = TimeSeries.from_dataframe(d, time_col="_t", value_cols=cov_cols)

    if len(target) != len(past_cov):
        raise RuntimeError(
            f"target and past_cov lengths differ: {len(target)} vs {len(past_cov)}"
        )

    return target, past_cov, cov_cols


def split_train_test(
    target: TimeSeries,
    past_cov: TimeSeries,
    test_size: float,
) -> tuple[TimeSeries, TimeSeries, TimeSeries, TimeSeries, int]:
    n = len(target)
    split_idx = int(n * (1.0 - test_size))

    train_t = target[:split_idx]
    test_t = target[split_idx:]
    train_c = past_cov[:split_idx]
    test_c = past_cov[split_idx:]
    return train_t, test_t, train_c, test_c, split_idx


def split_train_val(
    train_t: TimeSeries,
    train_c: TimeSeries,
    val_ratio: float,
    input_chunk_length: int,
    output_chunk_length: int,
) -> tuple[TimeSeries, TimeSeries, TimeSeries, TimeSeries]:
    n = len(train_t)
    val_len = int(round(n * float(val_ratio)))
    min_val = max(10, output_chunk_length + 2)
    if val_len < min_val:
        val_len = min_val

    train_len = n - val_len
    min_train = input_chunk_length + output_chunk_length + 5
    if train_len < min_train:
        train_len = min_train
        val_len = n - train_len

    if val_len < min_val:
        raise ValueError(
            f"Not enough data for validation split. n={n}, train_len={train_len}, val_len={val_len}."
        )

    tr_t = train_t[:train_len]
    va_t = train_t[train_len:]
    tr_c = train_c[:train_len]
    va_c = train_c[train_len:]
    return tr_t, va_t, tr_c, va_c


def build_model(cfg: TFTConfig) -> TFTModel:
    """
    Builds TFTModel with add_relative_index=True and CPU trainer.
    """
    if cfg.hidden_size % cfg.num_attention_heads != 0:
        raise ValueError(
            f"hidden_size must be divisible by num_attention_heads. Got {cfg.hidden_size} and {cfg.num_attention_heads}."
        )

    pl_trainer_kwargs = _pick_pl_trainer_kwargs(cfg)

    return TFTModel(
        input_chunk_length=cfg.input_chunk_length,
        output_chunk_length=cfg.output_chunk_length,
        hidden_size=cfg.hidden_size,
        lstm_layers=cfg.lstm_layers,
        num_attention_heads=cfg.num_attention_heads,
        dropout=cfg.dropout,
        batch_size=cfg.batch_size,
        n_epochs=cfg.epochs,
        optimizer_kwargs={"lr": cfg.lr, "weight_decay": cfg.weight_decay},
        add_relative_index=True,
        likelihood=None,
        loss_fn=nn.MSELoss(),
        random_state=cfg.seed,
        pl_trainer_kwargs=pl_trainer_kwargs,
        force_reset=True,
    )


def evaluate_one_step_ahead(
    model: TFTModel,
    target_s: TimeSeries,
    cov_s: TimeSeries | None,
    split_idx: int,
    scaler_target: Scaler | None = None,
    stride: int = 1,
    max_points: int | None = None,
) -> dict:
    """
    1-step ahead evaluation via historical_forecasts.
    """
    input_len = int(getattr(model, "input_chunk_length", 1) or 1)

    if max_points is not None and max_points > 0:
        start_candidate = max(split_idx, len(target_s) - int(max_points))
    else:
        start_candidate = split_idx

    start_pos = max(int(start_candidate), input_len)
    if start_pos >= len(target_s) - 1:
        raise RuntimeError(
            f"Not enough points for evaluation. start_pos={start_pos}, len(target)={len(target_s)}."
        )

    start_ts = target_s.time_index[start_pos]

    predict_kwargs = {}
    if cov_s is not None:
        predict_kwargs["past_covariates"] = cov_s

    forecasts = model.historical_forecasts(
        series=target_s,
        start=start_ts,
        start_format="value",
        forecast_horizon=1,
        stride=int(max(1, stride)),
        retrain=False,
        last_points_only=True,
        verbose=False,
        **predict_kwargs,
    )

    if forecasts is None or len(forecasts) == 0:
        raise RuntimeError("historical_forecasts() returned no forecasts.")

    if isinstance(forecasts, list):
        forecasts = concatenate(forecasts)

    actual = target_s.slice_intersect(forecasts)
    preds = forecasts.slice_intersect(actual)

    if scaler_target is not None:
        preds = scaler_target.inverse_transform(preds)
        actual = scaler_target.inverse_transform(actual)

    y_true = actual.values(copy=False).reshape(-1)
    y_pred = preds.values(copy=False).reshape(-1)

    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    n_ok = int(mask.sum())
    if n_ok == 0:
        raise RuntimeError("No valid points to score after alignment and forecasting.")

    y_true = y_true[mask]
    y_pred = y_pred[mask]

    mae = float(np.mean(np.abs(y_pred - y_true)))
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    return {"mae": mae, "rmse": rmse, "n_test_points": n_ok}


def compute_naive_metrics(df: pd.DataFrame) -> dict:
    y_true = df[TARGET_COLUMN].astype(float).to_numpy()
    y_pred = np.zeros_like(y_true, dtype=float)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    return {"mae": float(mae), "rmse": float(rmse)}


def save_artifact(
    output_dir: Path,
    model: TFTModel,
    cfg: TFTConfig,
    scaler_target: Scaler,
    scaler_cov: Scaler,
    covariate_columns: list[str],
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "tft_model"
    model.save(str(model_path))

    dump(
        {"scaler_target": scaler_target, "scaler_cov": scaler_cov},
        output_dir / "scalers.joblib",
    )

    metadata = {
        "cfg": asdict(cfg),
        "covariates": list(covariate_columns),
        "features_base": list(FEATURE_COLUMNS),
        "target": TARGET_COLUMN,
        "darts_model_path": str(model_path),
        "time_index": "range_index_steps",
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


def load_artifact(output_dir: Path) -> dict:
    output_dir = Path(output_dir)
    metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
    scalers = load(output_dir / "scalers.joblib")

    model_path = Path(metadata["darts_model_path"])
    model = TFTModel.load(str(model_path))

    return {"model": model, "metadata": metadata, "scalers": scalers}


def predict_next_return(df_raw: pd.DataFrame, artifact_dir: Path) -> float:
    """
    Predicts 1 step ahead at the end of the series.
    """
    art = load_artifact(artifact_dir)
    model: TFTModel = art["model"]
    scaler_target: Scaler = art["scalers"]["scaler_target"]
    scaler_cov: Scaler = art["scalers"]["scaler_cov"]

    df = df_raw.sort_values("open_time").reset_index(drop=True)
    target, past_cov, _ = build_timeseries(df)

    cov_s = scaler_cov.transform(past_cov)
    target_s = scaler_target.transform(target)

    pred_s = model.predict(n=1, series=target_s, past_covariates=cov_s, verbose=False)
    pred = scaler_target.inverse_transform(pred_s)
    return float(pred.values(copy=False).reshape(-1)[0])


def _ensure_cfg_valid(cfg: TFTConfig) -> None:
    if cfg.input_chunk_length < 2:
        raise ValueError("input_chunk_length must be >= 2")
    if cfg.output_chunk_length != 1:
        raise ValueError(
            "This trainer assumes output_chunk_length=1 for 1-step evaluation."
        )
    if cfg.hidden_size % cfg.num_attention_heads != 0:
        raise ValueError("hidden_size must be divisible by num_attention_heads.")


def _fit_and_score(
    cfg: TFTConfig,
    target_s: TimeSeries,
    cov_s: TimeSeries,
    split_idx: int,
    scaler_target: Scaler,
) -> dict:
    """
    Fits with internal train/val split from train portion and scores on test portion.
    """
    _ensure_cfg_valid(cfg)

    train_all_t = target_s[:split_idx]
    train_all_c = cov_s[:split_idx]

    min_train = cfg.input_chunk_length + cfg.output_chunk_length + 5
    if len(train_all_t) < min_train:
        raise ValueError(
            f"Not enough train data for chunks. need>={min_train}, have={len(train_all_t)}"
        )

    tr_t, va_t, tr_c, va_c = split_train_val(
        train_all_t,
        train_all_c,
        cfg.val_ratio,
        cfg.input_chunk_length,
        cfg.output_chunk_length,
    )

    _set_seed(cfg.seed)
    model = build_model(cfg)

    model.fit(
        series=tr_t,
        past_covariates=tr_c,
        val_series=va_t,
        val_past_covariates=va_c,
        verbose=False,
    )

    metrics = evaluate_one_step_ahead(
        model,
        target_s,
        cov_s,
        split_idx=split_idx,
        scaler_target=scaler_target,
        stride=cfg.eval_stride,
        max_points=cfg.eval_max_points,
    )

    return {"model": model, "metrics": metrics, "cfg": cfg}


def _make_grid_cfgs_full(base_cfg: TFTConfig, space: SearchSpace) -> list[TFTConfig]:
    """
    Builds the full cartesian grid with a divisibility constraint.
    """
    out: list[TFTConfig] = []
    for icl in space.input_chunk_length:
        for hs in space.hidden_size:
            for heads in space.num_attention_heads:
                if hs % heads != 0:
                    continue
                for nl in space.lstm_layers:
                    for do in space.dropout:
                        for lr in space.lr:
                            for bs in space.batch_size:
                                out.append(
                                    TFTConfig(
                                        input_chunk_length=int(icl),
                                        output_chunk_length=int(
                                            base_cfg.output_chunk_length
                                        ),
                                        hidden_size=int(hs),
                                        lstm_layers=int(nl),
                                        num_attention_heads=int(heads),
                                        dropout=float(do),
                                        batch_size=int(bs),
                                        epochs=int(base_cfg.epochs),
                                        lr=float(lr),
                                        weight_decay=float(base_cfg.weight_decay),
                                        test_size=float(base_cfg.test_size),
                                        val_ratio=float(base_cfg.val_ratio),
                                        seed=int(base_cfg.seed),
                                        eval_stride=int(base_cfg.eval_stride),
                                        eval_max_points=base_cfg.eval_max_points,
                                        early_stopping_patience=int(
                                            base_cfg.early_stopping_patience
                                        ),
                                        early_stopping_min_delta=float(
                                            base_cfg.early_stopping_min_delta
                                        ),
                                    )
                                )
    return out


def _make_grid_cfgs_random(
    base_cfg: TFTConfig, space: SearchSpace, trials: int
) -> list[TFTConfig]:
    """
    Samples random hyperparams (unique) with divisibility constraint.
    """
    out: list[TFTConfig] = []
    seen: set[tuple] = set()
    tries = 0
    max_tries = max(2000, trials * 100)

    while len(out) < trials and tries < max_tries:
        tries += 1
        icl = int(random.choice(space.input_chunk_length))
        hs = int(random.choice(space.hidden_size))
        heads = int(random.choice(space.num_attention_heads))
        if hs % heads != 0:
            continue
        nl = int(random.choice(space.lstm_layers))
        do = float(random.choice(space.dropout))
        lr = float(random.choice(space.lr))
        bs = int(random.choice(space.batch_size))

        key = (icl, hs, heads, nl, do, lr, bs)
        if key in seen:
            continue
        seen.add(key)

        out.append(
            TFTConfig(
                input_chunk_length=icl,
                output_chunk_length=int(base_cfg.output_chunk_length),
                hidden_size=hs,
                lstm_layers=nl,
                num_attention_heads=heads,
                dropout=do,
                batch_size=bs,
                epochs=int(base_cfg.epochs),
                lr=lr,
                weight_decay=float(base_cfg.weight_decay),
                test_size=float(base_cfg.test_size),
                val_ratio=float(base_cfg.val_ratio),
                seed=int(base_cfg.seed),
                eval_stride=int(base_cfg.eval_stride),
                eval_max_points=base_cfg.eval_max_points,
                early_stopping_patience=int(base_cfg.early_stopping_patience),
                early_stopping_min_delta=float(base_cfg.early_stopping_min_delta),
            )
        )

    if len(out) < trials:
        raise RuntimeError(
            f"Could not sample enough unique configs. requested={trials}, sampled={len(out)}"
        )

    return out


def _count_full_grid(space: SearchSpace) -> int:
    """
    Counts how many configs are in the full grid after constraints.
    """
    cnt = 0
    for icl in space.input_chunk_length:
        for hs in space.hidden_size:
            for heads in space.num_attention_heads:
                if hs % heads != 0:
                    continue
                for nl in space.lstm_layers:
                    for do in space.dropout:
                        for lr in space.lr:
                            for bs in space.batch_size:
                                cnt += 1
    return cnt


def run_single(cfg: TFTConfig, output_dir: Path) -> None:
    print("Loading dataset via random_forest_train.load_dataset() ...")
    df = load_rf_dataset(debug_samples=0)
    print(f"Dataset shape: {df.shape}")

    target, past_cov, cov_cols = build_timeseries(df)
    train_t, _, train_c, _, split_idx = split_train_test(
        target, past_cov, test_size=cfg.test_size
    )

    scaler_target = Scaler(StandardScaler())
    scaler_cov = Scaler(StandardScaler())
    scaler_target.fit(train_t)
    scaler_cov.fit(train_c)

    target_s = scaler_target.transform(target)
    cov_s = scaler_cov.transform(past_cov)

    print("\nTraining TFT (single run) ...")
    res = _fit_and_score(cfg, target_s, cov_s, split_idx, scaler_target=scaler_target)
    metrics = res["metrics"]
    model = res["model"]

    print("\nEvaluating (rolling 1-step on test) ...")
    print(
        f"  MAE={metrics['mae']:.6f} RMSE={metrics['rmse']:.6f} n_test_points={metrics['n_test_points']}"
    )

    save_artifact(
        output_dir, model, cfg, scaler_target, scaler_cov, covariate_columns=cov_cols
    )
    print(f"\nSaved artifact to: {output_dir}")

    naive = compute_naive_metrics(df)
    print("\nNaive baseline (ret_tomorrow = 0):")
    print(f"  MAE={naive['mae']:.6f} RMSE={naive['rmse']:.6f}")


def run_grid(
    base_cfg: TFTConfig, output_dir: Path, grid_mode: str, trials: int
) -> None:
    print("Loading dataset via random_forest_train.load_dataset() ...")
    df = load_rf_dataset(debug_samples=0)
    print(f"Dataset shape: {df.shape}")

    target, past_cov, cov_cols = build_timeseries(df)
    train_t, _, train_c, _, split_idx = split_train_test(
        target, past_cov, test_size=base_cfg.test_size
    )

    scaler_target = Scaler(StandardScaler())
    scaler_cov = Scaler(StandardScaler())
    scaler_target.fit(train_t)
    scaler_cov.fit(train_c)

    target_s = scaler_target.transform(target)
    cov_s = scaler_cov.transform(past_cov)

    space = SearchSpace()
    full_count = _count_full_grid(space)
    print(f"\nGrid space size (FULL): {full_count} trainings")

    if grid_mode == "random":
        cfgs = _make_grid_cfgs_random(base_cfg, space, trials=trials)
        print(f"Running grid mode RANDOM: {len(cfgs)} trainings")
    elif grid_mode == "full":
        cfgs = _make_grid_cfgs_full(base_cfg, space)
        print(f"Running grid mode FULL: {len(cfgs)} trainings")
    else:
        raise ValueError(
            f"Invalid GRID_MODE: {grid_mode}. Expected 'random' or 'full'."
        )

    print(
        f"  eval_stride={base_cfg.eval_stride} eval_max_points={base_cfg.eval_max_points}"
    )
    best = {"mae": float("inf"), "rmse": None, "cfg": None, "model": None, "n": None}

    for i, cfg_i in enumerate(cfgs, start=1):
        print(
            f"\n[{i}/{len(cfgs)}] "
            f"icl={cfg_i.input_chunk_length} hs={cfg_i.hidden_size} heads={cfg_i.num_attention_heads} "
            f"lstm={cfg_i.lstm_layers} do={cfg_i.dropout} lr={cfg_i.lr} bs={cfg_i.batch_size}"
        )
        try:
            res = _fit_and_score(
                cfg_i, target_s, cov_s, split_idx, scaler_target=scaler_target
            )
        except Exception as e:
            print(f"  -> SKIP (error): {e}")
            continue

        m = res["metrics"]
        print(
            f"  -> MAE={m['mae']:.6f} RMSE={m['rmse']:.6f} n_test_points={m['n_test_points']}"
        )

        if m["mae"] < best["mae"]:
            best = {
                "mae": m["mae"],
                "rmse": m["rmse"],
                "cfg": res["cfg"],
                "model": res["model"],
                "n": m["n_test_points"],
            }

    if best["model"] is None or best["cfg"] is None:
        raise RuntimeError("Grid search produced no valid model.")

    print("\n=== BEST TFT FROM GRID (by MAE) ===")
    print(f"  MAE={best['mae']:.6f} RMSE={best['rmse']:.6f} n_test_points={best['n']}")
    print(f"  cfg={best['cfg']}")

    save_artifact(
        output_dir,
        best["model"],
        best["cfg"],
        scaler_target,
        scaler_cov,
        covariate_columns=cov_cols,
    )
    print(f"\nSaved best artifact to: {output_dir}")

    naive = compute_naive_metrics(df)
    print("\nNaive baseline (ret_tomorrow = 0):")
    print(f"  MAE={naive['mae']:.6f} RMSE={naive['rmse']:.6f}")


def main() -> None:
    out_dir = Path(OUTPUT_DIR)

    base_cfg = TFTConfig(
        input_chunk_length=SINGLE_INPUT_CHUNK_LENGTH,
        output_chunk_length=SINGLE_OUTPUT_CHUNK_LENGTH,
        hidden_size=SINGLE_HIDDEN_SIZE,
        lstm_layers=SINGLE_LSTM_LAYERS,
        num_attention_heads=SINGLE_NUM_ATTENTION_HEADS,
        dropout=SINGLE_DROPOUT,
        batch_size=SINGLE_BATCH_SIZE,
        epochs=EPOCHS,
        lr=SINGLE_LR,
        weight_decay=WEIGHT_DECAY,
        test_size=TEST_SIZE,
        val_ratio=VAL_RATIO,
        seed=SEED,
        eval_stride=EVAL_STRIDE,
        eval_max_points=EVAL_MAX_POINTS,
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
        early_stopping_min_delta=EARLY_STOPPING_MIN_DELTA,
    )

    if RUN_MODE == "single":
        run_single(base_cfg, out_dir)
        return

    if RUN_MODE == "grid":
        run_grid(base_cfg, out_dir, grid_mode=GRID_MODE, trials=RANDOM_TRIALS)
        return

    raise ValueError(f"Invalid RUN_MODE: {RUN_MODE}. Expected 'single' or 'grid'.")


if __name__ == "__main__":
    main()
