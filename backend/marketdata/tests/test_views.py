import pytest
from rest_framework.test import APIClient
from rest_framework import status
import requests


@pytest.fixture
def api_client():
    return APIClient()


def test_binance_test_view_returns_success(api_client, mocker):
    mocker.patch(
        "marketdata.services.BinanceAPIService.BinanceAPIService.get_btc_price",
        return_value={"price": "50000"},
    )
    response = api_client.get("/marketdata/binance-test/")

    assert response.status_code == status.HTTP_200_OK
    assert response.data["price"] == "50000"


def test_historical_data_view_returns_bad_request_on_invalid_years_back(api_client):
    response = api_client.get("/marketdata/historical-data/", {"years_back": "invalid"})

    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert "years_back must be an integer" in response.data["error"]


def test_request_specific_data_view_requires_metrics_parameter(api_client):
    response = api_client.get("/marketdata/get-historical-data/")

    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert "metrics' is required" in response.data["error"]


def test_naive_model_view_returns_prediction(api_client, mocker):
    mocker.patch(
        "marketdata.services.ModelService.ModelService.naive", return_value=50000.0
    )
    response = api_client.get("/marketdata/naive-model/")

    assert response.status_code == status.HTTP_200_OK
    assert response.data["close_btc"] == 50000.0


def test_linear_regression_model_view_handles_missing_file(api_client, mocker):
    mocker.patch(
        "marketdata.services.ModelService.ModelService.linear_regression",
        side_effect=FileNotFoundError("Model not found"),
    )
    response = api_client.get("/marketdata/linear-regression-model/")

    assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    assert "Trained linear regression model not found" in response.data["error"]
