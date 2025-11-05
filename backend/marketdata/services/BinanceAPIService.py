import requests

class BinanceAPIService:
    BASE_URL = "https://api.binance.com/api/v3/"

    def __init__(self, timeout=5):
        self.timeout = timeout

    def _get(self, endpoint):
        url = self.BASE_URL + endpoint
        response = requests.get(url, timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def get_btc_price(self):
        data = self._get("ticker/price?symbol=BTCUSDT")
        return data