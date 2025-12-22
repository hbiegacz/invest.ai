# SHAP Analysis Report
> Ponieważ wartości, które model przewiduje są dość małe, dane zostały przeskalowane o **100**, aby na wykresach widzieć więcej szczegółów niż tylko +0.0 i -0.0.

## Eliminacja nie produktywnych cech

Poniżej znajduje się wykres podsumowania SHAP dla wszystkich wykorzystanych cech. Wykres składa się głównie z pionowych kresek – jest to kwestia działania drzew decyzyjnych w modelu Random Forest, które grupują dane i każda grupa dostaje tę samą wartość SHAP. Kilka zmiennych pojawia się **jako pojedyncza prosta pionowa linia** (np. `open_bnb` i `close_bnb` na samej górze wykresu). To wskazuje, że te niezależnie od tego, czy ich wartości są niskie (niebieski) czy wysokie (czerwony), model nie wie co z nimi zrobić itraktuje je tak samo. Wprowadzają szum  i powinny zostać wyeliminowane.


![All features shap summary](models/shap/shap_report_plots/image.png)

Cechy wytypowane do usunięcia to:
- surowe dane typu `open`, `close`, `high`, `low` --- model lepiej radzi sobie z feature'ami typu `ret_close_btc`, `ret_hl2_btc`, które nie pokazują poziomu ceny, a zmianę z dnia na dzień
- `hl2_eth` --- zbędne, ponieważ te same informacje są już zawarte w `ret_hl2_eth`
- `num_trades_btc`, `num_trades_eth`, `num_trades_bnb`, `num_trades_xrp` - osobne ilości transakcji dla wszystkich kryptowalut wprowadzają zbyt dużo szumu
- `gdp_lag1`


Poniżej znajduje się wykres podsumowania SHAP po wykluczeniu wcześniej zidentyfikowanych cech niepredykcyjnych. Udoskonalony model wykazuje poprawioną wydajność, osiągając niższy błąd średniokwadratowy (MSE) wynoszący 0.01681.

![Second shap summary](models/shap/shap_report_plots/image-1.png)

Możemy zauważyć, że nadal możemy usunąć kilka cech:
- `gdp`, `unrate`, `unrate_lag1` - aktualizowane zbyt rzadko w porównaniu do dziennych przewidywań cen crypto, przez co zawierają sporadyczne skoki wartości
- `roll_std_ret_close_btc_w21` 

## Even further analysis?

Na tym etapie wydaje się, że wyeliminowaliśmy wszelkie cechy, których model nie wydaje się rozumieć. To poprawiło wydajność modelu, ale nie znacząco, obecny MSE wynosi 0.01674.
![Third shap summary](models/shap/shap_report_plots/image-2.png)
