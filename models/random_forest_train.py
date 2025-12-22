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


# Allows manual override of the feature list
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
if "MANUAL_FEATURE_COLUMNS" in locals() and MANUAL_FEATURE_COLUMNS:
    FEATURE_COLUMNS = MANUAL_FEATURE_COLUMNS

GRID_MAX_DEPTH = [2, 3, 4, 5]
GRID_MIN_SAMPLES_LEAF = [10, 25, 50, 75, 100, 150, 200]
GRID_MAX_FEATURES = ["sqrt", "log2"]
GRID_N_ESTIMATORS = [200, 500]
GRID_MAX_SAMPLES = [0.3, 0.5, None]

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
