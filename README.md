# INVEST.AI

## Założenia oryginalne

Aplikcja wspomagająca podejmowanie decyzji inwestycyjnych dla świadomego inwestora, który śledzi wyniki finansowe spółek giełdowych. Konieczne jest stworzenie interfejsu wyświetlającego wykresy, stworzone na podstawie danych pobranych z zewnętrznej strony. System powinien spełniać dwie funkcje: po pierwsze informować użytkownika, jeśli któryś parametr wyników ulegnie znaczącej poprawie/pogorszeniu oraz po drugie, przy zastosowaniu modelu ML przewidywać kluczowe parametry wyników na kolejny kwartał.

## Nasza interpretacja

Szukając danych do treningu modelu, mieliśmy problemy ze znalezieniem aktualnych i live danych do tradycyjnej giełdy. Udało nam się natomiast odkryć, że giełda kryptowalut Binance oferuje w pełni darmowe API. Umożliwia ono nie tylko wyświetlenie potrzebnych danych w czasie rzeczywistym, ale też odtworzenie danych historycznych na kilkanaście lat wstecz.

Do tego, wszystkie API tradycyjnych giełd, dostępne za darmo lub w rozsądnej na projekt studencki cenie (max $10/miesiąc), oferują tylko ceny otwarcia i zamknięcia każdego dnia. Uznaliśmy, że zapewnienie uzytkownikowi danych in-real-time w dowolnym interwale jest po prostu ciekawsze i daje więcej możliwości.

Chemy też zmienić przewidywanie parametrów z wyników kolejnego kwartału na N świeczek w wybranym interwale. W przypadku kryptowalut podział na kwartały nie ma sensu, a 3 świeczki w interwale miesięcznym w pełni to zastąpią. Do tego, użytkownik może wybrać dowolny interwał.

# Bibliografia:

## API i dane historyczne

Postanowiliśmy wybrać API, które:

1. udostępnia dane w czasie rzeczywistym (czyli umożliwia narysowanie świeczki)
2. jest darmowe
3. umożliwi odtworzenie dokładnych danych historycznych

Idealne do tego celu wydaje się API Binance:
https://www.binance.com/api/v3

## Jak stworzyć plik z danymi historycznymi

API od Binance umożliwia wyświetlenie 1000 świeczek w wybranym interwale czasowym, do wybranego timestampa. To oznacza, że wywołując taki endpoint na przykład 5000 razy dla świeczek jednominutowych, uzyskujemy 10 lat dokładnych danych wykresu danej kryptowaluty. To wystarczy, żeby stworzyć dane do wytrenowania modelu do przewidywania ceny danej kryptowaluty.

# Implementacja

## Planowany stack technologiczny

-   React.js na frontend
-   Django na backend
-   SQLite na DB
-   Kilka modeli ML do przetestowania (więcej w sekcji o eksperymentach) - biblioteka TensorFlow + Sklearn
-   Docker
-   Jira

## Eksperymenty

Przetestowanie różnych modeli:

-   LSTM - potencjalnie najlepszy, ale też najtrudniejszy w implementacji
-   Random Forest - bardzo dobry do wielu zadań i stosunkowo prosty w implementacji
-   Logistic Regression - najprostszy w implemetacji

-   testowanie sposobu mierzenia jakości modelu (np miara oparta o rozkład normalny)

-   sprawdzenie różnych kryptowalut
-   sprawdzenie różnych interwałów
-   sprawdzenie wpływu BTC na inne kryptowaluty

-   cel skuteczności: powyżej 51%

## Planowana funkcjonalność

-   Kilka modeli przewidujących ceny kryptowaluty. Użytkownik może wybrać model poprzez dispatcher.
-   Rysowanie wykresu kryptowaluty + wybór interwału + dane analityczne (świeczki - informacje analityczne o zmianach)
-   Wybór docelowej krypotowaluty
-   Wyświetlanie przewidywanych N świeczek
-   Powiadomienia dla użytkownika
-   Konto użytkownika, z konfiguracją powiadomień

# Harmonogram

przed 7.11 - ukończenie prototypu (architektura, docker, szkielet aplikacji, dokumentacja i dokładna funkcjonalność)

Prototyp pod kątem funkcjonalnym: użytkownik na frontendzie wywołuje endpoint połączony przez nasz backend z Binance

## Lista zadań

-   zebranie i oczyszczenie danych
-   backend - stworzenie, logowanie użytkownika, wrappery endpointów Binance
-   backend - integracja API
-   backend - system powiadomień
-   baza danych - stworzenie struktury i tabel
-   frontend - interfejs
-   frontend - wyświetlanie wykresu
-   frontend - integracja modeli ML i LSTM
-   frontend - system powiadomień
-   wytrenowanie i dostosowanie modeli ML
-   wytrenowanie i dostosowanie LSTM

Zadania są podzielone na dwutygodniowe sprinty, opisane w pliku gantt_chart.png

## Pytania

-   Czy jest nam potrzebny cache? Przecież my opieramy się na live danych
