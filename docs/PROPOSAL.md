# INVEST.AI

## Założenia oryginalne

Aplikcja wspomagająca podejmowanie decyzji inwestycyjnych dla świadomego inwestora, który śledzi wyniki finansowe spółek giełdowych. Konieczne jest stworzenie interfejsu wyświetlającego wykresy, stworzone na podstawie danych pobranych z zewnętrznej strony. System powinien spełniać dwie funkcje: po pierwsze informować użytkownika, jeśli któryś parametr wyników ulegnie znaczącej poprawie/pogorszeniu oraz po drugie, przy zastosowaniu modelu ML przewidywać kluczowe parametry wyników na kolejny kwartał.

## Nasza interpretacja

Szukając danych do treningu modelu, mieliśmy problemy ze znalezieniem aktualnych datasetów do tradycyjnej giełdy. Udało nam się natomiast odkryć, że giełda kryptowalut Binance oferuje w pełni darmowe API. Umożliwia ono nie tylko wyświetlenie potrzebnych informacji w czasie rzeczywistym, ale też odtworzenie danych historycznych na kilkanaście lat wstecz.

Zastanowiliśmy się, czy istnieją fundamenty, mogące mieć wpływ na cenę bitcoina. Znaleźliśmy kilka parametrów, które spełniają ten warunek. Moglibyśmy stworzyć aplikację, która umożliwia porównanie, jaki wpływ dany potencjalny fundament ma na cenę bitcoina.

Wybrane przez nas potencjalne fundamenty do sprawdzenia to:

-   Siła nabywcza dolara
-   S&P 500
-   Aktywność on-chain (czyli jak dużo osób w danym momencie wykonuje transakcje krypto)

Sprawdzimy, czy występują zależności pomiędzy tymi potencjalnymi fundamentami, a ceną bitcoina. Użytkownik będzie mógł sam spojrzeć na ich czyste wykresy i szukać podobieństw. Do tego, miałby do dyspozycji 3 modele ML, który wyszukiwałby tego typu wzorce sam.

# Bibliografia:

## API i dane historyczne

Postanowiliśmy wybrać API, które:

1. są darmowe
2. umożliwią odtworzenie dokładnych danych historycznych

Idealne do tego celu wydają się api:
Binance
https://www.binance.com/api/v3

FRED API
https://fred.stlouisfed.org/

Stooq
https://stooq.com/

Coinmetrics:
https://docs.coinmetrics.io/api/v4/?utm_source=chatgpt.com

## Jak stworzyć pliki z danymi historycznymi?

API od Binance umożliwia wyświetlenie 1000 świeczek w wybranym interwale czasowym, do wybranego timestampa. To oznacza, że wywołując taki endpoint na przykład 5000 razy dla świeczek jednominutowych, uzyskujemy 10 lat dokładnych danych wykresu danej kryptowaluty.

Do tego, pozostałe API mają rozsądne limity, które też pozwolą wyświetlić nam aktualne dane.

# Implementacja

## Planowany stack technologiczny

-   React.js na frontend
-   Django na backend
-   SQLite na DB
-   Kilka modeli ML do przetestowania (więcej w sekcji o eksperymentach) - biblioteka TensorFlow + Sklearn oraz Darts
-   Docker
-   Jira

## Eksperymenty

Przetestowanie różnych modeli:

-   Linear Regression - jest najprostszy
-   Random Forest - bardzo dobry do wielu zadań i stosunkowo prosty w implementacji
-   TFT - potencjalnie najlepszy, ale też najtrudniejszy w implementacji

-   testowanie sposobu mierzenia jakości modelu (musimy zastanowić się nad sposobem mierzenia)

-   sprawdzenie wpływu fundamentów na cenę bitcoina

-   cel skuteczności: powyżej N%

## Planowana funkcjonalność

-   Wykresy ceny bitcoina
-   Wykresy nszych sprawdzanych parametrów
-   Model ML wyszukujący wpływ tych parametrów na cenę bitcoina
-   Działanie modelu zwizualizowane na frontendzie aplikacji

# Harmonogram

przed 7.11 - ukończenie prototypu (architektura, docker, szkielet aplikacji, dokumentacja i dokładna funkcjonalność)

Prototyp pod kątem funkcjonalnym: użytkownik na frontendzie wywołuje endpoint połączony przez nasz backend z każdym z api

## Lista zadań

-   zebranie i oczyszczenie danych
-   backend - stworzenie szkieletu backendu, wrappery endpointów Binance
-   backend - integracja pozostałych API
-   baza danych - stworzenie struktury i tabel
-   frontend - interfejs
-   frontend - wyświetlanie wykresów
-   frontend - integracja modeli ML i TFT
-   wytrenowanie i dostosowanie modeli ML
-   wytrenowanie i dostosowanie TFT
-   badanie, który fundament ma największy wpływ na cenę bitcoina
-   testy
-   instrukcja
-   dokumentacja końcowa

Zadania są podzielone na cztery sprinty, opisane w pliku gantt_chart.png:

![Gantt](gantt_chart.png)

## Harmonogram sprintów

-   Sprint 1: 24.10.2025 - 07.11.2025
-   Sprint 2: 08.11.2025 - 21.11.2025
-   Sprint 3: 22.11.2025 - 12.12.2025
-   Sprint 4: 13.12.2025 - 09.01.2026

## Pytania

-   Jak mierzyć błąd modelu ML przy badaniu wpływu jednego parametru na inny?
-   Jaki byłby rozsądny cel skuteczności? Mocno powiązane ze sposobem mierzenia.

## Plan na tworzenie modeli i podejście

Mamy już potrzebne dane: cena BTC, ETH, BNB, XRP, SPX, GDP, UNRATE, w kompatybilnym formacie danych.

### Ustalmy najpierw kilka ważnych szczegółów. Co będziemy przewidywać, jak to ocenimy i z czym to porównamy.

1. Będziemy przewidywać cenę BTC w dniu t na podstawie danych historycznych (t-n), gdzie n to liczba dni, na jakie patrzymy wstecz. Można go wywołać k razy, żeby przewidzieć na k dni do przodu.

2. Trzeba rozważyć, czy może zamiast przewidywać cenę, nie przewidzieć zwrotu z BTC? To może być odporne na wysokość ceny startowej i skupić się na wahaniach cen. (Albo log(zwrot), bo z jakiegoś powodu ML lubi logarytmy). Warto też sprawdzić kwadrat.

3. Będziemy porównywać predykcję z danymi prawdziwymi i na tej podstawie zmierzymy MAE albo RMSE. Gdyby model sobie nie radził, zawsze możemy przejść też na po prostu mierzenie kierunku (spadek czy wzrost). Można sprawdzać kierunek korygowany o prowizję (małe zmiany są tak małe, że kierunek nie ma znaczenia)

4. Na wykres świeczkowy nałożymy po prostu przewidywania modelu, żeby użytkownik mógł je łatwo zinterpretować.

### A oto modele, jakie utworzymy:

0. Naive model. Będzie po prostu przepisywał wartość z dnia poprzedniego. To będzie baseline, który musi być pobity przez każdy model.

1. Linear regression. Featurami będą historyczne wartości btc oraz wybranego parametru poukładane w sekwencje. Sprawdzimy różne parametry i zmierzymy sobie ich wpływ na trafność wyniku.

2. Random Forest. Zrobimy dla niego to samo co dla Linear Regression. Zakładam, że jego nieliniowość sprawi, że będzie znacznie lepszy, na jego podstawie dopasujemy format przekazywania modelowi danych, otrzymywanego wyniku i sposobu mierzenia błędu.

3. LSTM. Najprostszy i najlepiej udokumentowany model sekwencyjny - większe prawdopodobieństwo na to, że się uda, niż TFT.

4. TFT z użyciem biblioteki Darts. TFT podobno jest najlepszym modelem do radzenia sobie z sekwencjami, stąd wybór.

Notatki

-   dodawanie dodatkowych wykresów w trybie preview
-   korelacja
