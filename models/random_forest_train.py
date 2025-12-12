"""
Co pozmieniałem w tej wersji @Adrian:

1) Celem (target) nie jest cena BTC, tylko zwrot z następnego dnia:
   - liczymy log-return BTC: ret_btc_t = log(close_btc_t) - log(close_btc_{t-1})
   - target: ret_btc_next = ret_btc_t przesunięty o -1 (czyli zwrot "jutrzejszy")

   Dzięki temu model uczy się przewidywać względną zmianę, a nie poziom ceny (który jest niestacjonarny).

2) Featurey cenowe normalizujemy do log-returnów, zamiast karmić model poziomami:
   - dla każdej krypto (btc, eth, bnb, xrp) oraz spx bierzemy:
     a) ret_close_{asset}: log(close_t) - log(close_{t-1})
     b) ret_hl2_{asset}: log(hl2_t) - log(hl2_{t-1}),
        gdzie hl2 = (low + high) / 2

   To upraszcza rozkłady, stabilizuje wariancję i ogranicza wpływ samego trendu cenowego.

3) Featurey wolumenowe i aktywność rynkową agregujemy oraz stabilizujemy:
   - volume_sum = suma volume z (btc, eth, bnb, xrp, spx)
   - num_trades_sum = suma num_trades z (btc, eth, bnb, xrp)

   Następnie robimy z tego zmianę log(1+x):
   - dlog_volume_sum = log1p(volume_sum_t) - log1p(volume_sum_{t-1})
   - dlog_num_trades_sum = log1p(num_trades_sum_t) - log1p(num_trades_sum_{t-1})

   log1p diff jest odporny na duże skale i outliery i daje sensowny "procentowy" sygnał.

4) Makro (GDP, unrate) traktujemy ostrożnie:
   - te serie są publikowane rzadziej niż dziennie, więc dzienne pct_change daje prawie same zera i skoki.
   - dodatkowo, bez dat publikacji łatwo o leakage (przyszła wartość dostępna "za wcześnie").

   Dlatego:
   - bierzemy poziomy z lagu: gdp_lag1 = gdp.shift(1), unrate_lag1 = unrate.shift(1)
   - i dodajemy "sparse changes" liczone na lagach:
     gdp_growth = pct_change(gdp_lag1)
     unrate_change = unrate_lag1 - unrate_lag1.shift(1)

   To minimalna sensowna ochrona przed wstrzyknięciem przyszłości.

5) Regularyzacja po stronie modelu:
   - RandomForest sam w sobie ma regularyzację przez uśrednianie drzew,
     a dodatkowo ograniczamy złożoność hiperparametrami (max_depth, min_samples_leaf, max_features, max_samples).
   - Chronologiczny split (shuffle=False) utrzymuje realizm backtestu.

W praktyce ta normalizacja robi 2 rzeczy:
- usuwa niestacjonarność poziomów (ceny/volume rosną w długim terminie),
- ustawia wszystkie wejścia w podobnej semantyce: "zmiana względem wczoraj".

Debug:
- DEBUG_SAMPLES > 0 wypisze head/tail i statystyki featureów.
"""

"""
Moje wnioski (Adrian):
1. Testowanie jest teraz uczciwsze.
- wcześniej sprawdzaliśmy model tylko na jednym kawałku danych (ostatnie ~20%), więc mogliśmy trafić "łatwy" okres i wynik wyglądał lepiej niż w realu.
- teraz robimy 5 testów po kolei w czasie (walk-forward), więc widać jak model radzi sobie w różnych okresach.

2. Model ma tylko minimalną przewagę nad zgadywaniem "jutro = 0".
- średnio MAE/RMSE wychodzi odrobinę lepiej niż baseline, ale różnice są bardzo małe i często giną w tym, że różne okresy rynku są po prostu inne.
- czyli: jest lekki sygnał, ale nie jest to "mocny" model.

3. Z kierunkiem (czy jutro plus czy minus) jest trochę lepiej niż baseline.
- trafiamy znak ok. 52.7% vs 52.0% baseline.
- to mały plus, ale też nie działa równo w każdym foldzie (czasem jest prawie losowo).

4. Najwięcej informacji model bierze z innych rynków/krypto, a nie z makro.
- ważne są zwroty ETH/BNB/XRP i SPX + ich wygładzenia (EWM), oraz trochę cech BTC (np. hl2, zmienność).
- GDP/unrate w tej formie praktycznie nic nie wnoszą (część wygląda jak "zero sygnału").

5. Wywalanie całych grup cech prawie nic nie zmienia wyniku.
- jak usuniemy macro / volatility / ewm / spx, MAE pogarsza się tylko minimalnie.
- to znaczy: sygnał jest rozproszony i słaby, a model i tak głównie "kręci się" wokół krótkoterminowych returnów.

6. Uproszczenie (bez volume/trades) jest OK.
- usunięcie volume/trades nie pogorszyło jakości, a upraszcza model i zmniejsza ryzyko dopasowania do szumu.
"""

from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.inspection import permutation_importance


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "backend" / "data" / "historical_data.parquet"
DEFAULT_MODEL_PATH = ROOT_DIR / "backend" / "data" / "random_forest_btc.pkl"

CRYPTO_ASSETS = ["btc", "eth", "bnb", "xrp"]
INDEX_ASSETS = ["spx"]
ALL_ASSETS = CRYPTO_ASSETS + INDEX_ASSETS

TARGET_COLUMN = "ret_btc_next"

BASE_FEATURE_COLUMNS = [
    "ret_close_btc",
    "ret_hl2_btc",
    "ret_close_eth",
    "ret_hl2_eth",
    "ret_close_bnb",
    "ret_hl2_bnb",
    "ret_close_xrp",
    "ret_hl2_xrp",
    "ret_close_spx",
    "ret_hl2_spx",
    "dlog_volume_sum",
    "dlog_num_trades_sum",
    "gdp_lag1",
    "unrate_lag1",
    "gdp_growth",
    "unrate_change",
]

DROP_VOLUME_TRADES = True

EWM_SPANS = [7]
VOL_WINDOWS = [21]

EWM_FEATURE_COLUMNS = []
for span in EWM_SPANS:
    for a in ALL_ASSETS:
        EWM_FEATURE_COLUMNS.append(f"ewm_ret_close_{a}_s{span}")
        EWM_FEATURE_COLUMNS.append(f"ewm_ret_hl2_{a}_s{span}")
    EWM_FEATURE_COLUMNS.append(f"ewm_dlog_volume_sum_s{span}")
    EWM_FEATURE_COLUMNS.append(f"ewm_dlog_num_trades_sum_s{span}")
VOL_FEATURE_COLUMNS = [f"roll_std_ret_close_btc_w{w}" for w in VOL_WINDOWS]

FEATURE_COLUMNS = BASE_FEATURE_COLUMNS + EWM_FEATURE_COLUMNS + VOL_FEATURE_COLUMNS

if DROP_VOLUME_TRADES:
    FEATURE_COLUMNS = [
        c for c in FEATURE_COLUMNS
        if c not in ("dlog_volume_sum", "dlog_num_trades_sum")
        and not c.startswith("ewm_dlog_volume_sum_")
        and not c.startswith("ewm_dlog_num_trades_sum_")
    ]

GRID_MAX_DEPTH = [2, 3, 4, 5]
GRID_MIN_SAMPLES_LEAF = [10, 25, 50, 75, 100, 150, 200]
GRID_MAX_FEATURES = ["sqrt", "log2"]
GRID_N_ESTIMATORS = [200, 500]
GRID_MAX_SAMPLES = [0.3, 0.5, None]

# GRID_MAX_DEPTH = [3, 5, 7]
# GRID_MIN_SAMPLES_LEAF = [50, 100, 200, 500]
# GRID_MAX_FEATURES = ["sqrt", "log2"]
# GRID_N_ESTIMATORS = [50, 100]
# GRID_MAX_SAMPLES = [0.3, 0.5, None]

RUN_MODE = "grid"
TEST_SIZE = 0.2
DEBUG_SAMPLES = 0
OUTPUT_PATH = DEFAULT_MODEL_PATH

RF_CRITERION = "squared_error"
RF_MIN_SAMPLES_SPLIT = 2
RF_MIN_WEIGHT_FRACTION_LEAF = 0.0
RF_MAX_LEAF_NODES = None
RF_MIN_IMPURITY_DECREASE = 0.0
RF_BOOTSTRAP = True
RF_OOB_SCORE = False
RF_N_JOBS = -1
RF_RANDOM_STATE = 42
RF_VERBOSE = 0
RF_WARM_START = False
RF_CCP_ALPHA = 0.0
RF_MONOTONIC_CST_STR = None

SINGLE_RF_PARAMS = {
    "n_estimators": 100,
    "max_depth": 5,
    "min_samples_leaf": 200,
    "max_features": "log2",
    "max_samples": 0.3,
}


def _require_columns(df: pd.DataFrame, cols: list[str]) -> None:
    """
    Ensures required columns are present in the DataFrame.
    """
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in parquet: {missing}")


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

def parse_monotonic_cst(value: str | None) -> list[int] | None:
    """
    Parses monotonic_cst from comma-separated ints.
    Returns list[int] or None.
    """
    if value is None:
        return None

    value_str = str(value).strip()
    if not value_str or value_str.lower() == "none":
        return None

    parts = [p.strip() for p in value_str.split(",") if p.strip()]
    if not parts:
        return None

    return [int(p) for p in parts]


def load_dataset(debug_samples: int = 0) -> pd.DataFrame:
    """
    Loads parquet and builds normalized features + target ret_btc_next.
    """
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Parquet file not found at: {DATA_PATH}")

    df = pd.read_parquet(DATA_PATH, engine="pyarrow")

    if "open_time" in df.columns:
        df = df.sort_values("open_time").reset_index(drop=True)

    required = ["gdp", "unrate"]
    for a in ALL_ASSETS:
        required += [f"high_{a}", f"low_{a}", f"close_{a}", f"volume_{a}"]
    for a in CRYPTO_ASSETS:
        required += [f"num_trades_{a}"]

    _require_columns(df, required)

    for a in ALL_ASSETS:
        df[f"hl2_{a}"] = (df[f"low_{a}"] + df[f"high_{a}"]) / 2.0

    df["volume_sum"] = df[[f"volume_{a}" for a in ALL_ASSETS]].sum(axis=1)
    df["num_trades_sum"] = df[[f"num_trades_{a}" for a in CRYPTO_ASSETS]].sum(axis=1)

    for a in ALL_ASSETS:
        df[f"ret_close_{a}"] = _log_return(df[f"close_{a}"])
        df[f"ret_hl2_{a}"] = _log_return(df[f"hl2_{a}"])

    df["dlog_volume_sum"] = _dlog1p(df["volume_sum"])
    df["dlog_num_trades_sum"] = _dlog1p(df["num_trades_sum"])
    for span in EWM_SPANS:
        for a in ALL_ASSETS:
            df[f"ewm_ret_close_{a}_s{span}"] = _ewm_mean(df[f"ret_close_{a}"], span=span, shift=1)
            df[f"ewm_ret_hl2_{a}_s{span}"] = _ewm_mean(df[f"ret_hl2_{a}"], span=span, shift=1)

        df[f"ewm_dlog_volume_sum_s{span}"] = _ewm_mean(df["dlog_volume_sum"], span=span, shift=1)
        df[f"ewm_dlog_num_trades_sum_s{span}"] = _ewm_mean(df["dlog_num_trades_sum"], span=span, shift=1)
    
    for w in VOL_WINDOWS:
        df[f"roll_std_ret_close_btc_w{w}"] = _rolling_std(df["ret_close_btc"], window=w, shift=1)

    df["gdp_lag1"] = df["gdp"].shift(1)
    df["unrate_lag1"] = df["unrate"].shift(1)

    df["gdp_growth"] = df["gdp_lag1"].pct_change()
    df["unrate_change"] = df["unrate_lag1"] - df["unrate_lag1"].shift(1)

    df["ret_btc"] = _log_return(df["close_btc"])
    df[TARGET_COLUMN] = df["ret_btc"].shift(-1)

    df = df.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN]).reset_index(drop=True)

    if df.empty:
        raise ValueError("DataFrame is empty after preprocessing.")

    if debug_samples and debug_samples > 0:
        cols_to_show = []
        if "open_time" in df.columns:
            cols_to_show.append("open_time")
        cols_to_show += FEATURE_COLUMNS + [TARGET_COLUMN]

        print("\n=== DEBUG: feature samples (head) ===")
        print(df[cols_to_show].head(debug_samples).to_string(index=False))

        print("\n=== DEBUG: feature samples (tail) ===")
        print(df[cols_to_show].tail(debug_samples).to_string(index=False))

        print("\n=== DEBUG: feature stats ===")
        print(df[FEATURE_COLUMNS + [TARGET_COLUMN]].describe().T.to_string())

    return df


def build_model(**rf_kwargs):
    """
    Builds RandomForestRegressor instance from provided kwargs.
    """
    return RandomForestRegressor(**rf_kwargs)


def train_model(df: pd.DataFrame, test_size: float, rf_params: dict):
    """
    Trains RandomForestRegressor on FEATURE_COLUMNS to predict TARGET_COLUMN.
    Uses chronological train/test split (shuffle=False).
    """
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


def compute_naive_metrics(df: pd.DataFrame):
    """
    Computes metrics for naive baseline: always predict 0 return for tomorrow.
    """
    y_true = df[TARGET_COLUMN]
    y_pred = pd.Series(0.0, index=y_true.index)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5

    return {"mae": float(mae), "rmse": float(rmse)}


def save_model(model, output_path: Path):
    """
    Saves model payload (model, features list, target name) to .pkl file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "model": model,
        "features": FEATURE_COLUMNS,
        "target": TARGET_COLUMN,
        "target_column_in_df": TARGET_COLUMN,
    }

    dump(payload, output_path)


def print_metrics(metrics: dict) -> None:
    """
    Prints metrics in the exact format expected in logs.
    """
    print(f"  mae: {metrics['mae']}")
    print(f"  rmse: {metrics['rmse']}")
    print(f"  n_train: {metrics['n_train']}")
    print(f"  n_test: {metrics['n_test']}")


def base_rf_params() -> dict:
    """
    Returns RF params shared between single-run and grid.
    """
    return {
        "criterion": RF_CRITERION,
        "min_samples_split": RF_MIN_SAMPLES_SPLIT,
        "min_weight_fraction_leaf": RF_MIN_WEIGHT_FRACTION_LEAF,
        "max_leaf_nodes": RF_MAX_LEAF_NODES,
        "min_impurity_decrease": RF_MIN_IMPURITY_DECREASE,
        "bootstrap": RF_BOOTSTRAP,
        "oob_score": RF_OOB_SCORE,
        "n_jobs": RF_N_JOBS,
        "random_state": RF_RANDOM_STATE,
        "verbose": RF_VERBOSE,
        "warm_start": RF_WARM_START,
        "ccp_alpha": RF_CCP_ALPHA,
        "monotonic_cst": parse_monotonic_cst(RF_MONOTONIC_CST_STR),
    }

# def train_model_with_features(df: pd.DataFrame, feature_cols: list[str], test_size: float, rf_params: dict):
#     X = df[feature_cols].copy()
#     y = df[TARGET_COLUMN].copy()
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
#     model = build_model(**rf_params)
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)
#     mae = mean_absolute_error(y_test, y_pred)
#     rmse = mean_squared_error(y_test, y_pred) ** 0.5
#     metrics = {
#         "mae": float(mae),
#         "rmse": float(rmse),
#         "n_train": int(len(X_train)),
#         "n_test": int(len(X_test)),
#     }
#     return model, metrics, X_test, y_test


# def print_top_importances(model, feature_cols: list[str], X_test, y_test, top_k: int = 20) -> None:
#     mdi = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
#     print("\nTop feature importances (MDI):")
#     for name, val in mdi.head(top_k).items():
#         print(f"  {name}: {val:.6f}")
#     try:
#         pi = permutation_importance(
#             model, X_test, y_test,
#             n_repeats=5,
#             random_state=RF_RANDOM_STATE,
#             n_jobs=RF_N_JOBS,
#         )
#         perm = pd.Series(pi.importances_mean, index=feature_cols).sort_values(ascending=False)
#         print("\nTop feature importances (Permutation, mean over repeats):")
#         for name, val in perm.head(top_k).items():
#             print(f"  {name}: {val:.6f}")
#     except Exception as e:
#         print(f"\nPermutation importance skipped due to error: {e}")


# def run_group_ablation(df: pd.DataFrame, best_params: dict, test_size: float) -> None:
#     base_cols = FEATURE_COLUMNS
#     groups = {
#         "macro": ["gdp_lag1", "unrate_lag1", "gdp_growth", "unrate_change"],
#         "volatility": VOL_FEATURE_COLUMNS,
#         "ewm_all": EWM_FEATURE_COLUMNS,
#         "spx": [c for c in base_cols if "_spx" in c],
#         "alts": [c for c in base_cols if any(f"_{a}" in c for a in ["eth", "bnb", "xrp"])],
#         "volume_trades": (
#             ["dlog_volume_sum", "dlog_num_trades_sum"]
#             + [c for c in base_cols if c.startswith("ewm_dlog_volume_sum_") or c.startswith("ewm_dlog_num_trades_sum_")]
#         ),
#     }
#     _, base_metrics, _, _ = train_model_with_features(df, base_cols, test_size=test_size, rf_params=best_params)
#     print("\n=== ABLATION (same rows, drop groups) ===")
#     print(f"BASE (all features) mae={base_metrics['mae']:.10f} rmse={base_metrics['rmse']:.10f}")
#     for name, drop_cols in groups.items():
#         drop_cols = [c for c in drop_cols if c in base_cols]
#         keep_cols = [c for c in base_cols if c not in drop_cols]
#         _, m, _, _ = train_model_with_features(df, keep_cols, test_size=test_size, rf_params=best_params)
#         d_mae = m["mae"] - base_metrics["mae"]
#         d_rmse = m["rmse"] - base_metrics["rmse"]
#         print(f"\nDROP: {name}")
#         print(f"  n_drop={len(drop_cols)}")
#         print(f"  mae={m['mae']:.10f}  (delta {d_mae:+.10f})")
#         print(f"  rmse={m['rmse']:.10f} (delta {d_rmse:+.10f})")

# def time_series_cv_report(
#     df: pd.DataFrame,
#     feature_cols: list[str],
#     rf_params: dict,
#     n_splits: int = 5,
# ) -> dict:
#     X = df[feature_cols]
#     y = df[TARGET_COLUMN]
#     tss = TimeSeriesSplit(n_splits=n_splits)
#     fold_rows = []
#     for fold, (train_idx, test_idx) in enumerate(tss.split(X), start=1):
#         X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
#         y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
#         model = build_model(**rf_params)
#         model.fit(X_train, y_train)
#         y_pred = model.predict(X_test)
#         mae = mean_absolute_error(y_test, y_pred)
#         rmse = mean_squared_error(y_test, y_pred) ** 0.5
#         dir_acc = float(np.mean((y_pred >= 0) == (y_test.values >= 0)))
#         y0 = np.zeros_like(y_test.values, dtype=float)
#         mae0 = mean_absolute_error(y_test, y0)
#         rmse0 = mean_squared_error(y_test, y0) ** 0.5
#         dir0 = float(np.mean((y0 >= 0) == (y_test.values >= 0)))
#         fold_rows.append(
#             {
#                 "fold": fold,
#                 "n_train": int(len(train_idx)),
#                 "n_test": int(len(test_idx)),
#                 "mae": float(mae),
#                 "rmse": float(rmse),
#                 "dir_acc": float(dir_acc),
#                 "mae_baseline0": float(mae0),
#                 "rmse_baseline0": float(rmse0),
#                 "dir_acc_baseline0": float(dir0),
#             }
#         )
#     res = pd.DataFrame(fold_rows)
#     print("\n=== WALK-FORWARD CV (TimeSeriesSplit) ===")
#     print(res[["fold", "n_train", "n_test", "mae", "rmse", "dir_acc"]].to_string(index=False))
#     def _mean_std(col: str) -> tuple[float, float]:
#         return float(res[col].mean()), float(res[col].std(ddof=1)) if len(res) > 1 else (float(res[col].mean()), 0.0)
#     mae_m, mae_s = _mean_std("mae")
#     rmse_m, rmse_s = _mean_std("rmse")
#     dir_m, dir_s = _mean_std("dir_acc")
#     mae0_m, mae0_s = _mean_std("mae_baseline0")
#     rmse0_m, rmse0_s = _mean_std("rmse_baseline0")
#     dir0_m, dir0_s = _mean_std("dir_acc_baseline0")
#     print("\nSummary (mean ± std):")
#     print(f"  MAE : {mae_m:.10f} ± {mae_s:.10f}   (baseline0: {mae0_m:.10f} ± {mae0_s:.10f})")
#     print(f"  RMSE: {rmse_m:.10f} ± {rmse_s:.10f}   (baseline0: {rmse0_m:.10f} ± {rmse0_s:.10f})")
#     print(f"  DIR : {dir_m:.6f} ± {dir_s:.6f}       (baseline0: {dir0_m:.6f} ± {dir0_s:.6f})")
#     return {
#         "per_fold": res,
#         "summary": {
#             "mae_mean": mae_m,
#             "mae_std": mae_s,
#             "rmse_mean": rmse_m,
#             "rmse_std": rmse_s,
#             "dir_acc_mean": dir_m,
#             "dir_acc_std": dir_s,
#             "mae0_mean": mae0_m,
#             "mae0_std": mae0_s,
#             "rmse0_mean": rmse0_m,
#             "rmse0_std": rmse0_s,
#             "dir0_mean": dir0_m,
#             "dir0_std": dir0_s,
#         },
#     }


def run_grid_search(df: pd.DataFrame, test_size: float, output_path: Path) -> None:
    """
    Runs the full predefined hyper-grid, prints metrics for each combo,
    then prints and saves the best model (by MAE).
    """
    base_params = base_rf_params()

    total_combos = (
        len(GRID_MAX_DEPTH)
        * len(GRID_MIN_SAMPLES_LEAF)
        * len(GRID_MAX_FEATURES)
        * len(GRID_N_ESTIMATORS)
        * len(GRID_MAX_SAMPLES)
    )

    print(f"\nRunning full grid search: {total_combos} combinations")

    best_model = None
    best_params = None
    best_metrics = None
    best_mae = float("inf")

    combo_idx = 0
    for (
        max_depth,
        min_samples_leaf,
        max_features,
        n_estimators,
        max_samples,
    ) in product(
        GRID_MAX_DEPTH,
        GRID_MIN_SAMPLES_LEAF,
        GRID_MAX_FEATURES,
        GRID_N_ESTIMATORS,
        GRID_MAX_SAMPLES,
    ):
        combo_idx += 1
        params = {
            **base_params,
            "max_depth": max_depth,
            "min_samples_leaf": min_samples_leaf,
            "max_features": max_features,
            "n_estimators": n_estimators,
            "max_samples": max_samples,
        }

        print(f"\n[{combo_idx}/{total_combos}] params:")
        print(f"  max_depth: {max_depth}")
        print(f"  min_samples_leaf: {min_samples_leaf}")
        print(f"  max_features: {max_features}")
        print(f"  n_estimators: {n_estimators}")
        print(f"  max_samples: {max_samples}")

        model, metrics = train_model(df, test_size=test_size, rf_params=params)
        print_metrics(metrics)

        if metrics["mae"] < best_mae:
            best_mae = metrics["mae"]
            best_model = model
            best_params = params
            best_metrics = metrics

    if best_model is None or best_params is None or best_metrics is None:
        raise RuntimeError("Grid search did not produce any model.")
    
    # time_series_cv_report(df, FEATURE_COLUMNS, rf_params=best_params, n_splits=5)

    # best_model_refit, _, X_test, y_test = train_model_with_features(
    #     df, FEATURE_COLUMNS, test_size=test_size, rf_params=best_params
    # )
    # print_top_importances(best_model_refit, FEATURE_COLUMNS, X_test, y_test, top_k=25)
    # run_group_ablation(df, best_params=best_params, test_size=test_size)

    save_model(best_model, output_path)

    print("\n=== BEST MODEL FROM GRID SEARCH ===")
    print("params:")
    print(f"  max_depth: {best_params['max_depth']}")
    print(f"  min_samples_leaf: {best_params['min_samples_leaf']}")
    print(f"  max_features: {best_params['max_features']}")
    print(f"  n_estimators: {best_params['n_estimators']}")
    print(f"  max_samples: {best_params['max_samples']}")
    print("metrics:")
    print_metrics(best_metrics)
    print(f"\nSaved to: {output_path}")

    naive_metrics = compute_naive_metrics(df)
    print("\nNaive baseline (ret_tomorrow = 0):")
    print(f"  mae: {naive_metrics['mae']}")
    print(f"  rmse: {naive_metrics['rmse']}")


def run_single(df: pd.DataFrame, test_size: float, output_path: Path) -> None:
    """
    Trains a single RF config (SINGLE_RF_PARAMS), prints metrics and saves the model.
    """
    params = {
        **base_rf_params(),
        **SINGLE_RF_PARAMS,
    }

    print("\nTraining single config params:")
    print(f"  max_depth: {params.get('max_depth')}")
    print(f"  min_samples_leaf: {params.get('min_samples_leaf')}")
    print(f"  max_features: {params.get('max_features')}")
    print(f"  n_estimators: {params.get('n_estimators')}")
    print(f"  max_samples: {params.get('max_samples')}")

    model, metrics = train_model(df, test_size=test_size, rf_params=params)
    print("\nMetrics on test set:")
    print_metrics(metrics)

    save_model(model, output_path)
    print(f"\nSaved to: {output_path}")

    naive_metrics = compute_naive_metrics(df)
    print("\nNaive baseline (ret_tomorrow = 0):")
    print(f"  mae: {naive_metrics['mae']}")
    print(f"  rmse: {naive_metrics['rmse']}")


def main():
    """
    Entry point driven by constants (RUN_MODE, TEST_SIZE, DEBUG_SAMPLES, OUTPUT_PATH).
    """
    output_path = Path(OUTPUT_PATH)

    print(f"Loading dataset from: {DATA_PATH}")
    df = load_dataset(debug_samples=DEBUG_SAMPLES)
    print(f"Dataset shape after preprocessing: {df.shape}")

    if RUN_MODE == "grid":
        run_grid_search(df, test_size=TEST_SIZE, output_path=output_path)
        return

    if RUN_MODE == "single":
        run_single(df, test_size=TEST_SIZE, output_path=output_path)
        return

    raise ValueError(f"Invalid RUN_MODE: {RUN_MODE}. Expected 'grid' or 'single'.")


if __name__ == "__main__":
    main()
