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
]