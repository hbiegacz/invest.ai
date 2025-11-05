import requests

class CoinmetricsAPIService:
    BASE_URL = "https://community-api.coinmetrics.io/v4/"

    def __init__(self, timeout=5):
        self.timeout = timeout

    def _get(self, endpoint, params=None):
        url = self.BASE_URL + endpoint
        response = requests.get(url, params=params, timeout=self.timeout)
        response.raise_for_status()
        return response.json()
    
    def get_reference_rate(self):
        endpoint = "timeseries/asset-metrics"
        params = {
            "assets": "btc",
            "metrics": "ReferenceRateUSD",
            "frequency": "1d",
            "limit_per_asset": 1,
            "page_size": 1
        }
        data = self._get(endpoint, params=params)
        return data