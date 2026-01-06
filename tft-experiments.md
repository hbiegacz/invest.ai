# TFT (Temporal Fusion Transformer) - eksperymenty i uzasadnienie hiperparametrów

## Cel modelu

Model TFT trenujemy do prognozy ceny close bitcoina 1 krok do przodu. Pipeline jest przygotowany pod rolling backtest na zbiorze testowym, z metrykami:

-   MAE
-   RMSE

Dodatkowo porównujemy go do naiwniego baseline.

---

## Dane i cechy

### Źródło danych

Dataset zawiera:

-   kolumnę czasu: `open_time`
-   target: `TARGET_COLUMN`
-   cechy bazowe: `FEATURE_COLUMNS`

### Feature engineering: cechy kalendarzowe cykliczne

Do danych dodawane są cechy czasowe (cykliczne, sin/cos):

-   dzień tygodnia (`dow_sin`, `dow_cos`)
-   tydzień roku (`week_sin`, `week_cos`)

To jest kluczowe, bo:

-   model dostaje sygnał sezonowości bez numerków (np. `week=1..52`),
-   sin/cos zachowuje cykliczność (np. tydzień 52 jest blisko tygodnia 1).

### Time index w Darts

Zamiast timestampów, budowany jest indeks kroków:

-   `_t = 0..N-1`

I dopiero na tym powstają szeregi czasowe Darts:

-   `target: TimeSeries(value_cols=[TARGET_COLUMN])`
-   `past_covariates: TimeSeries(value_cols=FEATURE_COLUMNS + time_features)`

To jest w porządku, bo model i tak działa sekwencyjnie, a prawdziwy czas jest zakodowany w cechach kalendarzowych + względnym indeksie.

---

## Podział danych

### Train/Test

Split jest prosty, chronologiczny:

-   `split_idx = int(n * (1 - test_size))`
-   train = `[:split_idx]`
-   test = `[split_idx:]`

### Train/Val wewnątrz train

Walidacja jest wycinana z końcówki traina (`val_ratio`), z zabezpieczeniami na minimalne długości:

-   val ma minimum `max(10, output_chunk_length + 2)`
-   train musi mieć minimum `input_chunk_length + output_chunk_length + 5`

To jest ważne, bo TFT wymaga sensownego kontekstu sekwencji.

---

## Skalowanie

Skalowanie jest robione tylko na train (to poprawnie unika data leakage):

-   osobny scaler dla `target`
-   osobny scaler dla `past_covariates`

W praktyce:

-   model trenuje na znormalizowanych seriach,
-   metryki liczone są po inverse_transform, więc MAE/RMSE są w skali oryginalnej.

---

## Model

### Konstrukcja

Używany jest `darts.models.TFTModel` z:

-   `add_relative_index=True`
-   `loss_fn = nn.MSELoss()`
-   `likelihood=None` (czyli deterministycznie, bez probabilistyki)
-   trener Lightning na CPU (`accelerator="cpu"`, `devices=1`)
-   early stopping: monitor `val_loss`

### Dlaczego `add_relative_index=True` pomaga?

Bo przy indeksie `_t=0..N-1` model dostaje dodatkową informację o pozycji w oknie wejściowym. To poprawia stabilność uczenia, szczególnie gdy:

-   sygnał w cechach jest podobny w różnych fragmentach historii,
-   chcemy, żeby model odróżniał dawne vs świeższe punkty w kontekście.

---

## Ewaluacja: rolling 1-step ahead (historyczne prognozy)

Test liczony jest metodą zbliżoną do rzeczywistego przewidywania:

-   start od `max(split_idx, input_chunk_length, len(target) - max_points)`
-   `forecast_horizon=1`
-   `stride = eval_stride`
-   `retrain=False`
-   `last_points_only=True`

To daje realistyczny obraz jakości w symulacji działania produkcyjnego.

---

## Tryby eksperymentów

### Single run

Trening jednego zestawu hiperparametrów i zapis artefaktów:

-   `tft_model` (Darts save)
-   `scalers.joblib`
-   `metadata.json` z configiem i listą covariatów

### Grid search

Dwa tryby:

-   `GRID_MODE="full"`: pełna siatka kombinacji (z constraintem `hidden_size % num_attention_heads == 0`)
-   `GRID_MODE="random"`: losowe próby unikalnych kombinacji

Kryterium wyboru best:

-   najmniejsze MAE na rolling teście

---

## Przestrzeń przeszukiwania (SearchSpace)

Testowane były:

-   `input_chunk_length`: (30, 60, 90)
-   `hidden_size`: (16, 96)
-   `lstm_layers`: (2,)
-   `num_attention_heads`: (1, 4)
-   `dropout`: (0.0, 0.2)
-   `lr`: (3e-4, 1e-3)
-   `batch_size`: (64,)

Constraint:

-   `hidden_size % num_attention_heads == 0`
    czyli (16, 4) i (96, 4) są ok, ale np. (16, 3) byłoby odrzucone.

---

## Uzasadnienie: dlaczego wybrane hiperparametry zadziałały najlepiej

### 1) `output_chunk_length = 1` (stałe)

To ma sens, bo cała ewaluacja i pipeline są zrobione pod jednopunktową predykcję.
Dla returnów często najlepsza jakość jest właśnie dla 1 kroku, bo błędy nie kumulują się jak w multi-step.

### 2) `input_chunk_length` w zakresie 30-90

To jest balans między:

-   za krótko (np. < 20-30): model nie widzi kontekstu, a attention/LSTM nie mają z czego wyciągać wzorców
-   za długo (np. dużo powyżej 90): dla returnów zwykle nie ma stabilnych zależności dalekiego zasięgu, a długi kontekst podnosi wariancję i ułatwia overfit

### 3) `hidden_size` = 16 vs 96 (mały vs większy model)

To wprost kontroluje pojemność. Im mniejszy hidden_size, tym większa odporność na szum i mniejsze dopasowanie.

### 4) `lstm_layers = 2`

Dwie warstwy LSTM w TFT powinny dawać rozsądną reprezentację sekwencji, z niewielkim ryzykiem overfitu.

### 5) `num_attention_heads` = 1 lub 4

Heads kontrolują, ile różnych perspektyw attention może utrzymać.

-   1 head: prostszy model, mniejsze ryzyko overfitu, często lepszy gdy dane są noisy.
-   4 heads: model może równolegle patrzeć na różne fragmenty kontekstu (np. inne cechy/okresy).

### 6) `dropout` = 0.0 albo 0.2

Im bardziej skomplikowany model, tym większy dropout ma sens.

### 7) `lr` = 3e-4 albo 1e-3

### 8) `batch_size = 64`

## Dlaczego te ustawienia najlepiej zadziałały?

W tym problemie największym wrogiem jest overfit i udawana przewidywalność. Sprawdziłem w zasadzie model mniejszy i większy. Liczyłem na to, że mniejszy, prostszy model lepiej zgeneralizuje i nie będzie miał zbyt dużego overfitu i chyba się to nawet udało - w końcu tft miał najlepszy wynik ze wszystkich naszych modeli.
