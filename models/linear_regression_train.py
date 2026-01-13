import argparse
from pathlib import Path

import pandas as pd
from joblib import dump
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "backend" / "data" / "historical_data.parquet"
DEFAULT_MODEL_PATH = ROOT_DIR / "backend" / "data" / "linear_regression_btc.pkl"
FEATURE_COLUMNS = [
    "open_btc",
    "high_btc",
    "low_btc",
    "close_btc",
    "open_eth",
    "high_eth",
    "low_eth",
    "close_eth",
    "open_bnb",
    "high_bnb",
    "low_bnb",
    "close_bnb",
    "open_xrp",
    "high_xrp",
    "low_xrp",
    "close_xrp",
    "volume_btc",
    "volume_eth",
    "volume_bnb",
    "volume_xrp",
    "num_trades_btc",
    "num_trades_eth",
    "num_trades_bnb",
    "num_trades_xrp",
    "open_spx",
    "high_spx",
    "low_spx",
    "close_spx",
    "volume_spx",
    "gdp",
    "unrate",
]
TARGET_COLUMN = "ret_btc_next"


def load_dataset():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Parquet file not found at: {DATA_PATH}")
    df = pd.read_parquet(DATA_PATH, engine="pyarrow")
    if "open_time" in df.columns:
        df = df.sort_values("open_time").reset_index(drop=True)
    missing_features = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing_features:
        raise ValueError(f"Missing feature columns in parquet: {missing_features}")
    if "close_btc" not in df.columns:
        raise ValueError("Column 'close_btc' not found in parquet file.")
    df["ret_btc"] = df["close_btc"].pct_change()
    df[TARGET_COLUMN] = df["ret_btc"].shift(-1)
    df = df.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN])
    if df.empty:
        raise ValueError("DataFrame is empty after preprocessing.")
    return df


def build_pipeline(model_type="ols", alpha=1.0):
    if model_type == "ridge":
        reg = Ridge(alpha=alpha)
    elif model_type == "lasso":
        reg = Lasso(alpha=alpha)
    else:
        reg = LinearRegression()
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("regressor", reg),
        ]
    )


def train_model(df, test_size=0.2, model_type="ols", alpha=1.0):
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        shuffle=False,
    )
    pipeline = build_pipeline(model_type=model_type, alpha=alpha)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    metrics = {
        "mae": float(mae),
        "rmse": float(rmse),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }
    return pipeline, metrics


def compute_naive_metrics(df):
    y_true = df[TARGET_COLUMN]
    y_pred = pd.Series(0.0, index=y_true.index)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    return {"mae": float(mae), "rmse": float(rmse)}


def save_model(pipeline, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": pipeline,
        "features": FEATURE_COLUMNS,
        "target": TARGET_COLUMN,
        "target_column_in_df": TARGET_COLUMN,
    }
    dump(payload, output_path)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train linear model to predict next-day BTC return."
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data used for test set (default: 0.2).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_MODEL_PATH),
        help="Output path for trained model .pkl (default: models/linear_regression_btc.pkl).",
    )
    parser.add_argument(
        "--model-type",
        choices=["ols", "ridge", "lasso"],
        default="ols",
        help="Which linear model to use.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Regularization strength for Ridge/Lasso.",
    )
    return parser.parse_args()


def tune_lasso_alpha(df, test_size=0.2, alphas=None):
    if alphas is None:
        alphas = [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
    best_alpha = None
    best_mae = float("inf")
    best_metrics = None
    print("Tuning Lasso alpha...")
    for a in alphas:
        _, metrics = train_model(
            df,
            test_size=test_size,
            model_type="lasso",
            alpha=a,
        )
        mae = metrics["mae"]
        rmse = metrics["rmse"]
        print(f"  alpha={a:<8} -> MAE={mae:.6f}, RMSE={rmse:.6f}")
        if mae < best_mae:
            best_mae = mae
            best_alpha = a
            best_metrics = metrics
    print(f"\nBest alpha: {best_alpha} (MAE={best_mae:.6f})")
    return best_alpha, best_metrics


def main():
    args = parse_args()
    print(f"Loading dataset from: {DATA_PATH}")
    df = load_dataset()
    print(f"Dataset shape after preprocessing: {df.shape}")
    print("Training linear model...")
    if args.model_type == "lasso":
        best_alpha, _ = tune_lasso_alpha(df, test_size=args.test_size)
        args.alpha = best_alpha
    model, metrics = train_model(
        df,
        test_size=args.test_size,
        model_type=args.model_type,
        alpha=args.alpha,
    )
    output_path = Path(args.output)
    save_model(model, output_path)
    print(f"\nModel saved to: {output_path}")
    print("Metrics on test set:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    naive_metrics = compute_naive_metrics(df)
    print("\nNaive baseline (ret_tomorrow = 0):")
    for k, v in naive_metrics.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
