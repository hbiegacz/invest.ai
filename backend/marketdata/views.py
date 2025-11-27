import requests
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

from .services.BinanceAPIService import BinanceAPIService
from .services.FredAPIService import FredAPIService
from .services.CoinmetricsAPIService import CoinmetricsAPIService
from .services.StooqAPIService import StooqAPIService
from .services.HistoricalDataService import HistoricalDataService
from .services.DataReaderService import DataReaderService

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
        
class HistoricalDataView(APIView):
    def get(self, request, *args, **kwargs):
        years_back_param = request.query_params.get("years_back", "10")
        try:
            years_back = int(years_back_param)
        except ValueError:
            return Response(
                {"error": "years_back must be an integer"},
                status=status.HTTP_400_BAD_REQUEST,
            )
        filename = request.query_params.get("filename", "data.parquet")
        raw_excludes = request.query_params.getlist("exclude")
        excluded_cols = []
        for item in raw_excludes:
            parts = [c.strip() for c in item.split(",") if c.strip()]
            excluded_cols.extend(parts)
        parquet_flag = request.query_params.get("parquet", "").lower() in (
            "1",
            "true",
            "yes",
            "y",
        )
        service = HistoricalDataService()
        try:
            if parquet_flag:
                service.generate_parquet_file(
                    years_back=years_back,
                    filename=filename,
                    excluded_cols=excluded_cols or None,
                )
                return Response({"status": "SUCCESS"}, status=status.HTTP_200_OK)
            df = service.fetch_multi_symbol_df_excluding(
                years_back=years_back,
                excluded_cols=excluded_cols or None,
            )
            data = df.to_dict(orient="records")
            return Response(data, status=status.HTTP_200_OK)
        except Exception as e:
            return Response(
                {"error": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
            

class RequestSpecificDataView(APIView):
    """
    Endpoint for retrieving data form our historical data parquet file.
    Query parameters:
    - metrics: comma-separated list of columns (e.g., 'close_btc,volume_eth')
    - refresh: 'true' to force regeneration of the data file
    - years_back: optional integer number of years to go back when regenerating the file (default 10)
    
    Example: 
        /marketdata/get-historical-data/?metrics=close_btc,volume_btc,low_btc,open_btc
        /marketdata/get-historical-data/?metrics=close_btc,volume_btc,low_btc,open_btc&years_back=5
    """
    def get(self, request, *args, **kwargs):
        metrics_param = request.query_params.get("metrics", "")
        refresh_param = request.query_params.get("refresh", "").lower() in ("true", "1", "yes")
        years_back_param = request.query_params.get("years_back", None)

        if not metrics_param:
            return Response({"error": "Parameter 'metrics' is required (e.g. ?metrics=close_btc,volume_eth)"},
                            status=status.HTTP_400_BAD_REQUEST
            )

        requested_metrics = [m.strip() for m in metrics_param.split(",") if m.strip()]

        try:
            years_back = int(years_back_param) if years_back_param else 10
        except ValueError: 
            return Response({"error": "Parameter 'years_back' must be an integer."},
                            status=status.HTTP_400_BAD_REQUEST
            )
            
        try:
            service = DataReaderService()
            data = service.get_market_data( requested_metrics=requested_metrics, force_refresh=refresh_param, years_back=years_back)
            return Response(data, status=status.HTTP_200_OK)

        except ValueError as e:
            return Response( {"error": str(e)}, status=status.HTTP_400_BAD_REQUEST            )
        except RuntimeError as e:
            return Response({"error": str(e)}, status=status.HTTP_503_SERVICE_UNAVAILABLE)
        except Exception as e:
            return Response({"error": "An unexpected error occurred", "details": str(e)},
                            status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
