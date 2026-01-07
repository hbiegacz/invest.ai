# LSTM (Long Short-Term Memory) - dokumentacja modelu

## Cel modelu

Model LSTM prognozuje **1-step ahead log-return BTC na następny dzień**:

- `ret_btc_next = log(close_btc).diff().shift(-1)`

Nie przewidujemy poziomu ceny (niestacjonarnego), tylko **zwrot**. Na etapie inferencji zwrot przeliczamy na poziom ceny:

- `pred_close = last_close * exp(pred_ret)`

---

## Dane i cechy wejściowe

### Źródło danych

- `backend/data/historical_data.parquet`

Zawiera szeregi dla:

- krypto: BTC, ETH, BNB, XRP
- indeks: SPX
- makro: GDP, UNRATE

### Feature engineering (ten sam kierunek co RF/TFT)

Celem FE jest dostarczenie sygnałów opisujących **zmiany i trend**, zamiast surowych poziomów.

1. Zwroty cen (`ret_`)

- `ret_close_*` - log-return close
- `ret_hl2_*` - log-return `hl2 = (high + low)/2`

2. Aktywność rynku (`dlog`)

- `volume_sum` (suma wolumenów dla BTC/ETH/BNB/XRP/SPX)
- `num_trades_sum` (suma liczby transakcji dla BTC/ETH/BNB/XRP)
- `dlog_volume_sum = log1p(volume_sum).diff()`
- `dlog_num_trades_sum = log1p(num_trades_sum).diff()`

3. Wygładzanie trendu (`ewm`, span=7)

- `ewm_ret_close_*_s7`, `ewm_ret_hl2_*_s7`
- `ewm_dlog_volume_sum_s7`, `ewm_dlog_num_trades_sum_s7`

EWM liczone na serii opóźnionej (`shift=1`), żeby cechy w dniu _t_ korzystały wyłącznie z historii do _t-1_.

4. Makro (bez wycieku danych)

- `gdp_growth = (gdp.shift(1)).pct_change()`
- `unrate_change = unrate.shift(1) - unrate.shift(2)`

---

## Zestaw cech używany w modelu (26)

Model używa 26 kolumn (FEATURE_COLUMNS):

- `ret_close_{bnb, btc, eth, spx, xrp}`
- `ret_hl2_{bnb, btc, eth, spx, xrp}`
- `ewm_ret_close_{bnb, btc, eth, spx, xrp}_s7`
- `ewm_ret_hl2_{bnb, btc, eth, spx, xrp}_s7`
- `dlog_volume_sum`, `dlog_num_trades_sum`
- `ewm_dlog_volume_sum_s7`, `ewm_dlog_num_trades_sum_s7`
- `gdp_growth`, `unrate_change`

Target:

- `ret_btc_next`

---

## Budowa sekwencji (lookback window)

LSTM dostaje sekwencje o długości `lookback = L`:

- `X[i] = df[i : i+L, feature_cols]` -> shape `(L, n_features)`
- `y[i] = df[i+L-1, ret_btc_next]` (target przypisany do końca okna)

Interpretacja:

- "ostatnie **L dni** cech" -> "zwrot **jutro**".

---

## Split i preprocessing (bez leakage)

1. Sortowanie i czyszczenie

- dane sortowane po `open_time`
- usuwane wiersze z brakami po FE + budowie targetu

2. Podział train/val

- split chronologiczny: `train = [:split_idx]`, `val = [split_idx:]`
- brak shuffla (zachowujemy realizm szeregów czasowych)

3. Skalowanie cech (StandardScaler)

- `mean/std` liczone **wyłącznie na train**
- transformacja stosowana na train i val

4. Skalowanie targetu (włączone w finalnym runie)

- `ret_btc_next` standaryzowany na train (`z-score`)
- predykcja jest odskalowywana do skali zwrotu przed przeliczeniem na cenę

---

## Architektura

### Warstwa sekwencyjna

- `nn.LSTM(input_size=n_features, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)`
- dropout w LSTM aktywny tylko dla `num_layers > 1`

### Głowica regresyjna

- bierzemy wektor z ostatniego kroku: `out[:, -1, :]`
- MLP: `Linear -> ReLU -> Dropout -> Linear(..., 1)`
- wynik: pojedyncza wartość `pred_ret_btc_next`

---

## Trening (najważniejsze elementy)

- loss: `L1Loss` (raportujemy MAE)
- optymalizator: `AdamW(lr, weight_decay)`
- scheduler: `ReduceLROnPlateau` na `val_mae`
- gradient clipping: `grad_clip = 1.0`
- early stopping: zatrzymanie po `patience` epokach bez poprawy `val_mae`
- zapisywany jest najlepszy checkpoint (wg `val_mae`)

---

## Ostateczna wersja modelu (MAE, hiperparametry, wnioski)

### Konfiguracja finalnego runu

- `lookback = 96`, `seed = 314`
- `hidden_size = 64`, `num_layers = 2`, `dropout = 0.15`
- `batch_size = 64`
- `lr = 8e-4`, `weight_decay = 5e-4`
- `epochs = 100`, `patience = 15`
- `target_scaling = True`
- `n_features = 26`

### Wynik na walidacji (MAE)

- **Best val MAE: `0.015704`**

### Interpretacja przebiegu uczenia

- walidacja poprawia się głównie na początku treningu, a najlepszy wynik pojawia się wcześnie
- później widoczna jest stabilizacja/pogorszenie `val_mae`, mimo dalszego spadku `train_mae`

W praktyce oznacza to, że model szybko "wyciąga" dostępny sygnał, a dalsze uczenie zwiększa dopasowanie do danych treningowych bez poprawy uogólniania.

---

## Uzasadnienie doboru hiperparametrów

- `lookback=96`: kompromis między zbyt krótkim kontekstem (szum) a zbyt długim (wyższe ryzyko overfitu i niestabilności).
- `hidden_size=64`: umiarkowana pojemność; wystarczająca do modelowania relacji między wieloma cechami, bez nadmiernego "rozrostu" modelu.
- `num_layers=2`: daje większą ekspresję niż 1 warstwa, a jednocześnie jest bezpieczniejsze niż głębsze RNN w noisy danych.
- `dropout=0.15` + `weight_decay=5e-4`: regularizacja dobrana pod obserwację, że najlepsze `val_mae` pojawia się wcześnie (czyli potrzebujemy kontroli nad overfitem).
- `lr=8e-4` + `AdamW`: szybkie zejście z błędem w pierwszych epokach przy stabilnym treningu; scheduler dodatkowo "hamuje", gdy walidacja przestaje się poprawiać.
- `target_scaling=True` + `grad_clip=1.0`: stabilizacja treningu (skalowanie amplitudy celu + kontrola gradientów w LSTM).
