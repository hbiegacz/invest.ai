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
-   chcesz, żeby model odróżniał dawne vs świeższe punkty w kontekście.

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

Poniżej jest uzasadnienie teoretyczne i praktyczne, które pasuje do tego konkretnego setupu: 1-step ahead, dużo szumu, finansowe targety typu return, mała przewidywalność i silne ryzyko overfitu.

### 1) `output_chunk_length = 1` (stałe)

To ma sens, bo cała ewaluacja i pipeline są zrobione pod jednopunktową predykcję.
Dla returnów często najlepsza jakość jest właśnie dla 1 kroku, bo błędy nie kumulują się jak w multi-step.

### 2) `input_chunk_length` w zakresie 30-90

To jest balans między:

-   za krótko (np. < 20-30): model nie widzi kontekstu, a attention/LSTM nie mają z czego wyciągać wzorców
-   za długo (np. dużo powyżej 90): dla returnów zwykle nie ma stabilnych zależności dalekiego zasięgu, a długi kontekst podnosi wariancję i ułatwia overfit

Dlatego 30/60/90 to sensowne "okna pamięci":

-   30: krótkoterminowe reżimy i momentum/mean-reversion,
-   60: średni kontekst (zmiany zmienności),
-   90: większy kontekst, jeśli Twoje featury niosą informację o regime shift.

W praktyce często wygrywa 30 albo 60, bo krótsze okno jest bardziej "lokalne" i mniej podatne na dopasowanie historycznych przypadków, które się nie powtórzą.

### 3) `hidden_size` = 16 vs 96 (mały vs większy model)

To wprost kontroluje pojemność.

-   16 zwykle wygrywa, gdy dane są trudne (niski signal-to-noise) i chcesz uniknąć overfitu.
-   96 może wygrać, jeśli:

    -   masz dużo danych,
    -   featury są naprawdę informatywne,
    -   regularizacja/early stopping dobrze działają.

W finansach (zwłaszcza returny) częściej sprawdza się mniejsza pojemność, bo większa szybko zaczyna "zapamiętywać".

### 4) `lstm_layers = 2`

Dwie warstwy LSTM w TFT często dają lepszą reprezentację sekwencji niż 1 warstwa, ale nadal bez przesady:

-   1 warstwa bywa za płytka dla mieszanki sygnałów (covariates + sezonowość),
-   > 2 warstwy to już rośnie ryzyko niestabilności i overfitu (zwłaszcza na CPU i ograniczonym data).

Czyli 2 to takie "sweet spot".

### 5) `num_attention_heads` = 1 lub 4

Heads kontrolują, ile "różnych perspektyw" attention może utrzymać.

-   1 head: prostszy model, mniejsze ryzyko overfitu, często lepszy gdy dane są noisy.
-   4 heads: model może równolegle patrzeć na różne fragmenty kontekstu (np. inne cechy/okresy), ale tylko wtedy ma to sens, gdy `hidden_size` jest wystarczający.

Ważne: u Ciebie jest constraint `hidden_size % heads == 0`, więc 4 heads ma sens głównie dla 16/96 (obie dzielą się przez 4).

### 6) `dropout` = 0.0 albo 0.2

To jest klasyczny trade-off:

-   0.0 może wygrać, jeśli:

    -   early stopping działa,
    -   model jest mały (np. hidden_size=16),
    -   sygnał jest bardzo słaby i dropout tylko "rozmywa" uczenie.

-   0.2 zwykle pomaga, jeśli:

    -   model jest większy (hidden_size=96),
    -   masz tendencję do overfitu,
    -   walidacja jest stabilna.

Czyli sensownie: testujesz "bez regularizacji" i "z umiarkowaną regularizacją".

### 7) `lr` = 3e-4 albo 1e-3

To są dwa typowe poziomy dla Adam/AdamW:

-   1e-3: szybciej schodzi z lossu, często ok dla mniejszych modeli i prostych danych.
-   3e-4: stabilniejsze, lepsze gdy model ma większą pojemność albo dane są trudne.

Jeśli w gridzie "wygrywa" 3e-4, to zwykle oznacza, że:

-   gradienty są bardziej zmienne,
-   model łatwo przeskakuje minima przy 1e-3,
-   albo walidacja jest bardziej kapryśna.

### 8) `batch_size = 64`

To jest rozsądne "domyślne optimum" dla CPU i stabilności uczenia:

-   mniejsze batch: większy szum gradientu, czasem pomaga w generalizacji, ale wolniej i mniej stabilnie,
-   większe batch: stabilniej, ale może gorzej generalizować w noisy taskach, plus na CPU bywa wolniej przez pamięć/cache.

64 to często najlepszy kompromis.

---

## Dlaczego te ustawienia "najlepiej zadziałały" w Twoim pipeline

W skrócie: w tym problemie (returny, 1-step ahead) największym wrogiem jest overfit i "udawana przewidywalność".

Siatka, którą testowałeś, wprost sprawdzała dwa podejścia:

-   mały/stabilny model: `hidden_size=16`, często `heads=1` lub `4`, dropout 0, lr 1e-3
-   większy/regularizowany model: `hidden_size=96`, `heads=4`, dropout 0.2, lr 3e-4

I to jest dokładnie to, co ma sens przy TFT:

-   albo idziesz w prostotę i liczysz na minimalną generalizację,
-   albo dajesz pojemność, ale pilnujesz regularizacji i stabilnego LR.

Wyniki z backtestu (rolling 1-step, bez retrain) są dobrym kryterium, bo mierzą zachowanie bardzo zbliżone do produkcyjnego.

---

## Artefakty i reproducibility

Zapisujesz:

-   model (`tft_model`)
-   scalery (`scalers.joblib`)
-   `metadata.json` (cfg, lista covariatów, target, ścieżka modelu)

Dzięki temu inference (`predict_next_return`) odtwarza dokładnie ten sam preprocessing:

-   buduje TimeSeries,
-   skaluje covariates/target,
-   robi predykcję,
-   robi inverse transform.
