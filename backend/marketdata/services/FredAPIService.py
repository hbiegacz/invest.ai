import os
import requests
from dotenv import load_dotenv

load_dotenv()


class FredAPIService:
    BASE_URL = "https://api.stlouisfed.org/fred/"

    def __init__(self, timeout=5):
        self.api_key = os.getenv("FRED_API_KEY")
        if not self.api_key:
            raise ValueError("FRED_API_KEY is not set in environment variables")
        self.timeout = timeout

    def _get(self, endpoint, params=None):
        if params is None:
            params = {}
        params["api_key"] = self.api_key
        params["file_type"] = "json"
        url = self.BASE_URL + endpoint
        response = requests.get(url, params=params, timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def get_basic_economic_data(self):
        endpoint = "series/observations"
        unemploy_data = self._get(
            endpoint, params={"series_id": "UNRATE", "sort_order": "desc", "limit": 1}
        )
        gdp_data = self._get(
            endpoint, params={"series_id": "GDP", "sort_order": "desc", "limit": 1}
        )

        latest_unemploy = (
            unemploy_data["observations"][0] if unemploy_data["observations"] else {}
        )
        latest_gdp = gdp_data["observations"][0] if gdp_data["observations"] else {}

        return {
            "country": "USA",
            "latest_unemployment_rate": latest_unemploy.get("value", "N/A"),
            "unemployment_date": latest_unemploy.get("date", "N/A"),
            "latest_gdp": latest_gdp.get("value", "N/A"),
            "gdp_date": latest_gdp.get("date", "N/A"),
        }
