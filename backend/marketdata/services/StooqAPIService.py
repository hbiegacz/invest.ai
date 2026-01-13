import requests
import csv
import io


class StooqAPIService:
    BASE_URL = "https://stooq.com/q/l/"

    def __init__(self, timeout=5):
        self.timeout = timeout

    def _get(self, endpoint, params=None):
        url = self.BASE_URL + endpoint
        response = requests.get(url, params=params, timeout=self.timeout)
        response.raise_for_status()
        return response.text

    def get_sp500_data(self):
        endpoint = ""
        params = {
            "s": "^spx",  # symbol S&P500 index
            "f": "sd2t2ohlcv",  # fields: symbol, date, time, open, high, low, close, volume
            "h": "",  # no header override
            "e": "csv",  # response format CSV
        }
        csv_data = self._get(endpoint, params=params)
        csv_reader = csv.DictReader(io.StringIO(csv_data))
        data = [row for row in csv_reader]
        return data
