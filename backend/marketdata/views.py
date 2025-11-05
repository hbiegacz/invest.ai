import requests
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

from .services.BinanceAPIService import BinanceAPIService
from .services.FredAPIService import FredAPIService
from .services.CoinmetricsAPIService import CoinmetricsAPIService
from .services.StooqAPIService import StooqAPIService


class BinanceTestView(APIView):
    def get(self, request, *args, **kwargs):
        service = BinanceAPIService()
        try:
            data = service.get_btc_price()
            return Response(data, status=status.HTTP_200_OK)
        except requests.RequestException as e:
            return Response({"error": "Could not fetch price from Binance API", "details": str(e)},
                            status=status.HTTP_503_SERVICE_UNAVAILABLE)

class FredTestView(APIView):
    def get(self, request, *args, **kwargs):
        service = FredAPIService()
        try:
            data = service.get_basic_economic_data()
            return Response(data, status=status.HTTP_200_OK)
        except requests.RequestException as e:
             return Response({"error": "Could not fetch data from Stooq API", "details": str(e)},
                            status=status.HTTP_503_SERVICE_UNAVAILABLE)
        
class StooqTestView(APIView):
    def get(self, request, *args, **kwargs):
        service = StooqAPIService()
        try:
            data = service.get_sp500_data()
            return Response(data, status=status.HTTP_200_OK)
        except requests.RequestException as e:
            return Response({"error": "Could not fetch data from Stooq API", "details": str(e)},
                            status=status.HTTP_503_SERVICE_UNAVAILABLE)

class CoinmetricsTestView(APIView):
    def get(self, request, *args, **kwargs):
        service = CoinmetricsAPIService()
        try:
            data = service.get_reference_rate()
            return Response(data, status=status.HTTP_200_OK)
        except requests.RequestException as e:
            return Response({"error": "Could not fetch data from Coinmetrics API", "details": str(e)},
                            status=status.HTTP_503_SERVICE_UNAVAILABLE)