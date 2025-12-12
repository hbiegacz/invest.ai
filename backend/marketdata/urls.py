from django.urls import path
from . import views

urlpatterns = [
    path('binance-test/', views.BinanceTestView.as_view(), name='binance-test'),
    path('fred-test/', views.FredTestView.as_view(), name='fred-test'),
    path('stooq-test/', views.StooqTestView.as_view(), name='stooq-test'),
    path('coinmetrics-test/', views.CoinmetricsTestView.as_view(), name='coinmetrics-test'),
    path('historical-data/', views.HistoricalDataView.as_view(), name='historical-data'),
    path('get-historical-data/', views.RequestSpecificDataView.as_view(), name='get-historical-data'),
    path('naive-model/', views.NaiveModelView.as_view(), name='naive-model'),
    path('linear-regression-model/', views.LinearRegressionModelView.as_view(), name='linear-regression-model'),
    path('random-forest-model/', views.RandomForestModelView.as_view(), name='random-forest-model'),
    path('lstm-model/', views.LSTMModelView.as_view(), name='lstm-model'),
    path('tft-model/', views.TFTModelView.as_view(), name='tft-model'),
]