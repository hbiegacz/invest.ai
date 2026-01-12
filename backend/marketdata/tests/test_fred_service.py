import pytest
import requests
import os
from marketdata.services.FredAPIService import FredAPIService

def test_fred_service_raises_value_error_if_api_key_missing(mocker):
    mocker.patch.dict(os.environ, {"FRED_API_KEY": ""})
    with pytest.raises(ValueError, match="FRED_API_KEY is not set"):
        FredAPIService()

def test_fred_service_returns_combined_economic_data_on_success(mocker):
    mocker.patch.dict(os.environ, {"FRED_API_KEY": "fake_key"})
    
    def mock_get(url, params, timeout):
        mock_resp = mocker.Mock()
        mock_resp.status_code = 200
        if params.get("series_id") == "UNRATE":
            mock_resp.json.return_value = {"observations": [{"value": "3.5", "date": "2024-01-01"}]}
        elif params.get("series_id") == "GDP":
            mock_resp.json.return_value = {"observations": [{"value": "25000", "date": "2024-01-01"}]}
        return mock_resp

    mocker.patch("requests.get", side_effect=mock_get)

    service = FredAPIService()
    result = service.get_basic_economic_data()

    assert result["latest_unemployment_rate"] == "3.5"
    assert result["latest_gdp"] == "25000"
    assert result["country"] == "USA"

def test_fred_service_handles_empty_observations(mocker):
    mocker.patch.dict(os.environ, {"FRED_API_KEY": "fake_key"})
    
    mock_resp = mocker.Mock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"observations": []}
    mocker.patch("requests.get", return_value=mock_resp)

    service = FredAPIService()
    result = service.get_basic_economic_data()

    assert result["latest_unemployment_rate"] == "N/A"
    assert result["latest_gdp"] == "N/A"
