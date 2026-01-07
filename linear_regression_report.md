# Linear Regression (OLS / Ridge / Lasso) - dokumentacja modelu

## Cel modelu

Model liniowy służy do prognozy **zwrotu BTC na następny dzień** (1-step ahead):

- `ret_btc_next = pct_change(close_btc).shift(-1)`

Czyli w dniu _t_ używamy cech z dnia _t_, aby przewidzieć zwrot w dniu _t+1_.

---

## Dane i przygotowanie datasetu

### Źródło danych

- `backend/data/historical_data.parquet`

Po wczytaniu dane są sortowane chronologicznie po `open_time` (jeśli kolumna istnieje).

### Budowa targetu

W kodzie target jest tworzony następująco:

- `ret_btc = close_btc.pct_change()`
- `ret_btc_next = ret_btc.shift(-1)`

Następnie usuwane są wiersze z brakami w cechach lub w target (`dropna`).

---

## Cechy wejściowe (FEATURE_COLUMNS)

W tym modelu używamy **surowych poziomów** (OHLC, wolumen, liczba transakcji, indeks SPX oraz makro).

Ceny (OHLC) dla krypto:

- `open_btc`, `high_btc`, `low_btc`, `close_btc`
- `open_eth`, `high_eth`, `low_eth`, `close_eth`
- `open_bnb`, `high_bnb`, `low_bnb`, `close_bnb`
- `open_xrp`, `high_xrp`, `low_xrp`, `close_xrp`

Wolumen i liczba transakcji:

- `volume_btc`, `volume_eth`, `volume_bnb`, `volume_xrp`
- `num_trades_btc`, `num_trades_eth`, `num_trades_bnb`, `num_trades_xrp`

Indeks SPX:

- `open_spx`, `high_spx`, `low_spx`, `close_spx`, `volume_spx`

Makro:

- `gdp`, `unrate`

Target:

- `ret_btc_next`

---

## Split danych (train/test)

Podział jest chronologiczny, bez losowego mieszania:

- `test_size = 0.2`
- `shuffle = False`

Dla uruchomionego runu:

- `n_train = 1680`
- `n_test = 420`

To utrzymuje realistyczny scenariusz predykcji w szeregach czasowych (trenujemy na przeszłości i testujemy na przyszłości).

---

## Pipeline i skalowanie

Model jest trenowany jako `sklearn.pipeline.Pipeline`:

1. `StandardScaler()`
2. regresor liniowy: `LinearRegression` / `Ridge` / `Lasso`

Skalowanie jest szczególnie istotne dla modeli z regularyzacją (Ridge/Lasso), bo współczynniki kary zależą od skali cech.

---

## Warianty modelu

Skrypt obsługuje 3 tryby (`--model-type`):

### 1) OLS (LinearRegression)

- klasyczna regresja najmniejszych kwadratów bez regularyzacji

### 2) Ridge

- regresja z karą L2: `Ridge(alpha=alpha)`
- stabilizuje współczynniki i zmniejsza wariancję, zwykle pomaga przy współliniowości cech

### 3) Lasso

- regresja z karą L1: `Lasso(alpha=alpha)`
- może zerować część wag (selekcja cech), co bywa korzystne przy dużej liczbie skorelowanych sygnałów

---

## Strojenie hiperparametrów (Lasso)

Jeśli `--model-type lasso`, wykonywane jest proste strojenie `alpha` na zbiorze testowym (chronologicznym):

Testowane wartości:

- `[0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]`

Wybór "best":

- minimalne `MAE` na zbiorze testowym

Z logów:

- **Best alpha: `0.005`**
- **MAE = `0.016855`**
- **RMSE = `0.023352`**

---

## Metryki

Raportowane metryki na zbiorze testowym:

- `MAE = mean_absolute_error(y_test, y_pred)`
- `RMSE = sqrt(mean_squared_error(y_test, y_pred))`

---

## Wyniki eksperymentów (z logów)

### OLS (LinearRegression)

- **MAE:** `0.043481953958960734`
- **RMSE:** `0.0649079753374987`
- `n_train=1680`, `n_test=420`

W tej konfiguracji model liniowy bez regularyzacji nie generalizuje dobrze na test-set.

### Lasso (z tuningiem alpha)

- **MAE:** `0.01685521013312673`
- **RMSE:** `0.0233521064909458`
- najlepsze `alpha = 0.005`

Widać, że regularyzacja L1 znacząco poprawia wynik względem OLS.

---

## Artefakty i zapis modelu

Model jest zapisywany do:

- `backend/data/linear_regression_btc.pkl`

Format zapisu (`joblib.dump`) zawiera słownik:

- `model`: pipeline (scaler + regressor)
- `features`: lista `FEATURE_COLUMNS`
- `target`: `ret_btc_next`
- `target_column_in_df`: `ret_btc_next`

To umożliwia spójne odtworzenie inferencji (te same cechy w tej samej kolejności + ten sam scaler).
