# Random Forest

## Zbiór danych

Rozpoczynając pracę nad random forestem korzystaliśmy z danych historycznych, które zawierały następujące wartości:

#### Ceny otwarte, najwyższe, najniższe, zamknięte

-   open_btc, open_eth, open_bnb, open_xrp, open_spx
-   high_btc, high_eth, high_bnb, high_xrp, high_spx
-   low_btc, low_eth, low_bnb, low_xrp, low_spx
-   close_btc, close_eth, close_bnb, close_xrp, close_spx

#### Wolumen i liczba transakcji

-   volume_btc, volume_eth, volume_bnb, volume_xrp, volume_spx
-   num_trades_btc, num_trades_eth, num_trades_bnb, num_trades_xrp, num_trades_spx

#### Indeksy globalne

-   gdp
-   unrate

## Budowa modelu

Rozpoczynając pracę nad modelem, zdecydowaliśmy, że najważniejszymi parametrami naszego modelu będą:

-   `max_depth` - ogranicza głębokość drzewa
-   `min_samples_leaf` - minimalna liczba próbek w liściu
-   `max_features` - liczba cech, które bierze pod uwagę pojedyncze drzewo
-   `n_estimators` - liczba drzew w lesie
-   `max_samples` - liczba próbek, na których uczy się każde drzewo
-   `criterion` - funkcja oceniająca jakość podziału

Dlatego właśnie to optymalizacją tych parametrów zajęliśmy się w pierwszej kolejności.

## Pierwsze wnioski i problemy

Wstępna optymalizacja metodą **Grid Search** wykazała niepokojące tendencje. Model dążył do tworzenia skrajnie płytkich lasów, co sugerowało wysoki poziom szumu w danych.

```bash
=== BEST MODEL FROM GRID SEARCH ===
Best hyperparameters:
  max_depth: 3
  min_samples_leaf: 300
  max_features: log2
  n_estimators: 50
  max_samples: 0.5
```

Zaniepokoiły nas następujące rzeczy:

-   **Płytkie drzewa** --- Drzewa o maksymalnej głębokości 3 (`max_depth: 3`), nie pozwalają na wykrycie złożonych zależności i reguł między danymi.
-   **Mało drzew** --- Większa liczba drzew (`n_estimators: 50`) nie przynosi poprawy, co sugeruje, że dane są mocno zaszumione.
-   **Duża liczba próbek na liściach** --- Wysoka liczba próbek (`min_samples_leaf: 300`) sprawia, że las mocno uśrednia predykcje i nie wykrywa nowych, krótkoterminowych trendów.

W związku z tym, zdecydowaliśmy, że problem stanowią wykorzystywane przez nas dane, które wprowadzają szum do naszego modelu.

## Przetworzenie i dodanie nowych cech

Aby wyeliminować szum i dostarczyć modelowi bardziej czytelne sygnały, dokonaliśmy transformacji danych, wprowadzając nowe grupy cech:

#### Zwroty z cen `ret_`

-   `ret_close_*` - dzienna zmiana ceny zamknięcia danego rynku (np. BTC, ETH, SPX) wyrażona jako różnica między dniem bieżącym, a poprzednim.
-   `ret_hl_*` - analogiczny wskaźnik liczony dla ceny uśrednionej z danego dnia (średnia z wartości high i low).

#### Aktywność na rynku `dlog`

-   `dlog_volume_sum` - zmiana łącznego wolumenu obrotu na wszystkich rynkach w skali logarytmicznej (dzień do dnia).
-   `dlog_num_trades_sum` - odpowiednik powyższej miary dla łącznej liczby transakcji na rynku kryptowalut.

#### Wartości wygładzone `ewm`

Czyli trend w aktywności rynku zamiast dziennego szumu.

-   `ewm_ret_close_*` i `ewm_ret_hl2_*` - wykładniczo ważona średnia zwrotów cen, gdzie nowsze obserwacje mają większą wagę niż starsze (lepsze odwzorowanie aktualnego trendu).
-   `ewm_dlog_volume_sum_*` i `ewm_dlog_num_trades_sum_*` - identyczna metoda wygładzania zastosowana dla zmian wolumenu i liczby transakcji.

#### Zmienność `roll_std`

-   `roll_std_ret_close_btc_*` - miara zmienności rynku, liczona jako odchylenie standardowe zwrotów BTC w ruchomym oknie czasowym (np. 7, 21 dni).

#### Makro `gdp`, `unrate`

-   `gdp_lag1`, `gdp_growth`, `unrate_lag1`, `unrate_change` - poziomy i zmiany PKB oraz bezrobocia z poprzednich okresów. Używamy wartości opóźnionych (lag), ponieważ dane te publikowane są rzadziej. Dzięki temu chronimy model przed wyciekiem danych z przyszłości (data leakage).

## Analiza shap

Za pomocą narzędzia SHAP przyjrzeliśmy się poszczególnym cechom naszego modelu.

Na poniższym wykresie przedstawiono 20 cech, które mają największy wpływ na predykcję drzew. Wykres składa się głównie z pionowych kresek - jest to kwestia działania drzew decyzyjnych w modelu Random Forest, które grupują dane i każda grupa dostaje tę samą wartość SHAP. Kilka zmiennych pojawia się **jako pojedyncza prosta pionowa linia** (np. `open_bnb` i `close_bnb` na samej górze wykresu). To wskazuje, że te niezależnie od tego, czy ich wartości są niskie (niebieski) czy wysokie (czerwony), model nie wie co z nimi zrobić i traktuje je tak samo. Wprowadzają szum i powinny zostać wyeliminowane.

![All features shap summary](models/shap/shap_report_plots/image.png)

W pierwszej kolejności rzucała się w oczy konieczność eliminacji surowych danych giełdowych, takich jak `open_{asset}`, `high_{asset}`, `low_{asset}`, `close_{asset}`, `volume_{asset}`, `num_trades_{asset}`. Zostały one zamienione lepszymi cechami, obrazującymi zmianę wartości, a nie surowe poziomy.

Kolejnymi cechami, które nie okazały się produktywne są dane makroekonomiczne, takie jak `gdp`, `unrate`, `unrate_lag1`.

Po eliminacji tych cech, tak prezentuje się wykres podsumowania SHAP:
![Third shap summary](models/shap/shap_report_plots/image-2.png)

## Ostateczna wersja modelu

Finalnie postanowiliśmy skupić się na danych opisujących **zmianę** cen (`ret_close_` czy `ret_hl2_`), zamiast surowych wartości. Wykorzystaliśmy także cechy wygładzone (`ewm_ret_close_` czy `ewm_ret_hl2_`) i zsumowane dla różnych walut (`dlog_num_trades_sum`, `dlog_volume_sum`). Ostateczny model uzyskał na zbiorze testowym wynik `MAE ≈ 0.01679` oraz `RMSE ≈ 0.02330`.

Lista cech:

-   `ret_close_bnb`,
-   `ret_close_btc`,
-   `ret_close_eth`,
-   `ret_close_spx`,
-   `ret_close_xrp`,
-   `ret_hl2_bnb`,
-   `ret_hl2_btc`,
-   `ret_hl2_eth`,
-   `ret_hl2_spx`,
-   `ret_hl2_xrp`,
-   `ewm_dlog_num_trades_sum_s7`,
-   `ewm_dlog_volume_sum_s7`,
-   `ewm_ret_close_bnb_s7`,
-   `ewm_ret_close_btc_s7`,
-   `ewm_ret_close_eth_s7`,
-   `ewm_ret_close_spx_s7`,
-   `ewm_ret_close_xrp_s7`,
-   `ewm_ret_hl2_bnb_s7`,
-   `ewm_ret_hl2_btc_s7`,
-   `ewm_ret_hl2_eth_s7`,
-   `ewm_ret_hl2_spx_s7`,
-   `ewm_ret_hl2_xrp_s7`,
-   `dlog_num_trades_sum`,
-   `dlog_volume_sum`,
-   `gdp_growth`,
-   `unrate_change`,
-   `ret_btc`

A także parametrów:

```python
n_estimators=100,
max_depth=5,
min_samples_leaf=200,
max_features="log2",
max_samples=0.3,
```
