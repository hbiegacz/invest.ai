# FEATURE_COLUMNS = [
#     "open_btc",
#     "high_btc",
#     "low_btc",
#     "close_btc",
#     "open_eth",
#     "high_eth",
#     "low_eth",
#     "close_eth",
#     "open_bnb",
#     "high_bnb",
#     "low_bnb",
#     "close_bnb",
#     "open_xrp",
#     "high_xrp",
#     "low_xrp",
#     "close_xrp",
#     "volume_btc",
#     "volume_eth",
#     "volume_bnb",
#     "volume_xrp",
#     "num_trades_btc",
#     "num_trades_eth",
#     "num_trades_bnb",
#     "num_trades_xrp",
#     "open_spx",
#     "high_spx",
#     "low_spx",
#     "close_spx",
#     "volume_spx",
#     "gdp",
#     "unrate",
# ]


# - max_depth
# - min_samples_leaf
# - max_features
# - n_estimators
# - max_samples

# - criterion


# - n_jobs dla paralelizmu

# - wyłączanie featurów

# trening 1 - wszystko, default  mae: 0.020019812835128284 rmse: 0.026178989947603936

import argparse
from pathlib import Path

import pandas as pd
from joblib import dump
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "backend" / "data" / "historical_data.parquet"
DEFAULT_MODEL_PATH = ROOT_DIR / "backend" / "data" / "random_forest_btc.pkl"

FEATURE_COLUMNS = [
    # "open_btc",
    # "high_btc",
    # "low_btc",
    "close_btc",
    # "open_eth",
    # "high_eth",
    # "low_eth",
    "close_eth",
    # "open_bnb",
    # "high_bnb",
    # "low_bnb",
    "close_bnb",
    # "open_xrp",
    # "high_xrp",
    # "low_xrp",
    "close_xrp",
    # "volume_btc",
    # "volume_eth",
    # "volume_bnb",
    # "volume_xrp",
    # "num_trades_btc",
    # "num_trades_eth",
    # "num_trades_bnb",
    # "num_trades_xrp",
    # "open_spx",
    # "high_spx",
    # "low_spx",
    # "close_spx",
    # "volume_spx",
    # "gdp",
    # "unrate",
]

TARGET_COLUMN = "ret_btc_next"


def load_dataset():
    """
    Ładuje dane z pliku parquet, sortuje po open_time jeśli jest,
    liczy dzienny zwrot BTC i tworzy target ret_btc_next (zwrot z kolejnego dnia).
    Sprawdza brakujące kolumny i wyrzuca błędy jeśli czegoś brakuje.
    """
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


def build_model(**rf_kwargs):
    """
    Buduje instancję RandomForestRegressor na podstawie przekazanych parametrów.
    """
    return RandomForestRegressor(**rf_kwargs)


def train_model(df, test_size=0.2, rf_params=None):
    """
    Trenuje model RandomForestRegressor na danych wejściowych,
    zwraca wytrenowany model oraz słownik z metrykami MAE, RMSE i rozmiarami zbiorów.
    """
    if rf_params is None:
        rf_params = {}

    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        shuffle=False,
    )

    model = build_model(**rf_params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5

    metrics = {
        "mae": float(mae),
        "rmse": float(rmse),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }

    return model, metrics


def compute_naive_metrics(df):
    """
    Liczy metryki dla naiwnej strategii, w której przewidywany zwrot jutra to zawsze 0.
    """
    y_true = df[TARGET_COLUMN]
    y_pred = pd.Series(0.0, index=y_true.index)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5

    return {"mae": float(mae), "rmse": float(rmse)}


def save_model(model, output_path):
    """
    Zapisuje model wraz z listą feature'ów i nazwą kolumny celu do pliku .pkl.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "model": model,
        "features": FEATURE_COLUMNS,
        "target": TARGET_COLUMN,
        "target_column_in_df": TARGET_COLUMN,
    }

    dump(payload, output_path)


def str_to_bool(value):
    """
    Pomocnicza funkcja do parsowania wartości bool z argparse.
    Akceptuje m.in. true, false, yes, no, 1, 0.
    """
    if isinstance(value, bool):
        return value

    value_lower = value.lower()

    if value_lower in {"true", "t", "yes", "y", "1"}:
        return True
    if value_lower in {"false", "f", "no", "n", "0"}:
        return False

    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_optional_number_or_str(value):
    """
    Parsuje wartość, która może być None, liczbą lub stringiem.
    Używane dla max_features i max_samples.
    """
    if value is None:
        return None

    value_str = str(value).strip()
    if value_str.lower() == "none":
        return None

    try:
        if "." in value_str:
            return float(value_str)
        return int(value_str)
    except ValueError:
        return value_str


def parse_monotonic_cst(value):
    """
    Parsuje monotonic_cst podane jako string z liczbami oddzielonymi przecinkami.
    Zwraca listę int albo None.
    """
    if value is None:
        return None

    value_str = str(value).strip()
    if not value_str:
        return None

    parts = [p.strip() for p in value_str.split(",") if p.strip()]
    if not parts:
        return None

    return [int(p) for p in parts]


def parse_args():
    """
    Parsuje argumenty CLI, w tym wszystkie istotne parametry RandomForestRegressor.
    """
    parser = argparse.ArgumentParser(
        description="Train RandomForestRegressor to predict next-day BTC return."
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
        help="Output path for trained model .pkl (default: backend/data/random_forest_btc.pkl).",
    )

    parser.add_argument(
        "--n-estimators",
        type=int,
        default=100,
        help="Number of trees in the forest.",
    )

    parser.add_argument(
        "--criterion",
        type=str,
        default="squared_error",
        choices=["squared_error", "absolute_error", "friedman_mse", "poisson"],
        help="Function to measure the quality of a split.",
    )

    parser.add_argument(
        "--max-depth",
        type=int,
        default=None,
        help="Maximum depth of the tree. Default None means nodes are expanded until all leaves are pure.",
    )

    parser.add_argument(
        "--min-samples-split",
        type=int,
        default=2,
        help="Minimum number of samples required to split an internal node.",
    )

    parser.add_argument(
        "--min-samples-leaf",
        type=int,
        default=1,
        help="Minimum number of samples required to be at a leaf node.",
    )

    parser.add_argument(
        "--min-weight-fraction-leaf",
        type=float,
        default=0.0,
        help="Minimum weighted fraction of the sum of weights required to be at a leaf node.",
    )

    parser.add_argument(
        "--max-features",
        type=str,
        default="1.0",
        help="Number of features to consider when looking for the best split. "
             "Can be 'sqrt', 'log2', 'None' lub liczba (np. 0.5).",
    )

    parser.add_argument(
        "--max-leaf-nodes",
        type=int,
        default=None,
        help="Grow trees with at most max_leaf_nodes in best-first fashion.",
    )

    parser.add_argument(
        "--min-impurity-decrease",
        type=float,
        default=0.0,
        help="A node will be split if this split induces a decrease of the impurity >= this value.",
    )

    parser.add_argument(
        "--bootstrap",
        type=str_to_bool,
        default=True,
        help="Whether bootstrap samples are used when building trees.",
    )

    parser.add_argument(
        "--oob-score",
        type=str_to_bool,
        default=False,
        help="Whether to use out-of-bag samples to estimate R^2 on unseen data.",
    )

    parser.add_argument(
        "--n-jobs",
        type=int,
        default=None,
        help="Number of jobs to run in parallel. None means 1, -1 means all processors.",
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=None,
        help="Random seed for reproducibility.",
    )

    parser.add_argument(
        "--verbose",
        type=int,
        default=0,
        help="Controls the verbosity when fitting and predicting.",
    )

    parser.add_argument(
        "--warm-start",
        type=str_to_bool,
        default=False,
        help="If True, reuse the solution of the previous call to fit and add more estimators.",
    )

    parser.add_argument(
        "--ccp-alpha",
        type=float,
        default=0.0,
        help="Complexity parameter used for Minimal Cost-Complexity Pruning.",
    )

    parser.add_argument(
        "--max-samples",
        type=str,
        default=None,
        help="If bootstrap is True, the number of samples to draw from X to train each base estimator. "
             "Can be int, float, or None.",
    )

    parser.add_argument(
        "--monotonic-cst",
        type=str,
        default=None,
        help="Optional monotonic constraints as comma separated ints, e.g. '1,0,-1'.",
    )

    return parser.parse_args()


def main():
    """
    Główna funkcja: ładuje dane, trenuje RandomForestRegressor, zapisuje model i wypisuje metryki.
    """
    args = parse_args()

    print(f"Loading dataset from: {DATA_PATH}")
    df = load_dataset()
    print(f"Dataset shape after preprocessing: {df.shape}")

    max_features = parse_optional_number_or_str(args.max_features)
    max_samples = parse_optional_number_or_str(args.max_samples)
    monotonic_cst = parse_monotonic_cst(args.monotonic_cst)

    rf_params = {
        "n_estimators": args.n_estimators,
        "criterion": args.criterion,
        "max_depth": args.max_depth,
        "min_samples_split": args.min_samples_split,
        "min_samples_leaf": args.min_samples_leaf,
        "min_weight_fraction_leaf": args.min_weight_fraction_leaf,
        "max_features": max_features,
        "max_leaf_nodes": args.max_leaf_nodes,
        "min_impurity_decrease": args.min_impurity_decrease,
        "bootstrap": args.bootstrap,
        "oob_score": args.oob_score,
        "n_jobs": args.n_jobs,
        "random_state": args.random_state,
        "verbose": args.verbose,
        "warm_start": args.warm_start,
        "ccp_alpha": args.ccp_alpha,
        "max_samples": max_samples,
        "monotonic_cst": monotonic_cst,
    }

    print("Training RandomForestRegressor...")
    model, metrics = train_model(
        df,
        test_size=args.test_size,
        rf_params=rf_params,
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
