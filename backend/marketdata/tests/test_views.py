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


def test_binance_test_view_handles_exception(api_client, mocker):
    mocker.patch(
        "marketdata.services.BinanceAPIService.BinanceAPIService.get_btc_price",
        side_effect=requests.RequestException("API Error"),
    )
    response = api_client.get("/marketdata/binance-test/")

    assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
    assert "Could not fetch price from Binance API" in response.data["error"]


def test_fred_test_view_returns_success(api_client, mocker):
    mocker.patch(
        "marketdata.services.FredAPIService.FredAPIService.get_basic_economic_data",
        return_value={"data": "economic"},
    )
    response = api_client.get("/marketdata/fred-test/")
    assert response.status_code == status.HTTP_200_OK


def test_fred_test_view_handles_exception(api_client, mocker):
    mocker.patch(
        "marketdata.services.FredAPIService.FredAPIService.get_basic_economic_data",
        side_effect=requests.RequestException("API Error"),
    )
    response = api_client.get("/marketdata/fred-test/")
    assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE


def test_stooq_test_view_handles_exception(api_client, mocker):
    mocker.patch(
        "marketdata.services.StooqAPIService.StooqAPIService.get_sp500_data",
        side_effect=requests.RequestException("API Error"),
    )
    response = api_client.get("/marketdata/stooq-test/")
    assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE


def test_coinmetrics_test_view_handles_exception(api_client, mocker):
    mocker.patch(
        "marketdata.services.CoinmetricsAPIService.CoinmetricsAPIService.get_reference_rate",
        side_effect=requests.RequestException("API Error"),
    )
    response = api_client.get("/marketdata/coinmetrics-test/")
    assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE


def test_historical_data_view_returns_bad_request_on_invalid_years_back(api_client):
    response = api_client.get("/marketdata/historical-data/", {"years_back": "invalid"})

    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert "years_back must be an integer" in response.data["error"]


def test_historical_data_view_parquet_flag(api_client, mocker):
    mock_service = mocker.patch(
        "marketdata.views.HistoricalDataService", autospec=True
    ).return_value
    response = api_client.get(
        "/marketdata/historical-data/", {"parquet": "yes", "years_back": "5"}
    )

    assert response.status_code == status.HTTP_200_OK
    assert response.data["status"] == "SUCCESS"
    mock_service.generate_parquet_file.assert_called_once()


def test_historical_data_view_handles_exception(api_client, mocker):
    mocker.patch(
        "marketdata.services.HistoricalDataService.HistoricalDataService.fetch_multi_symbol_df_excluding",
        side_effect=Exception("Generic Error"),
    )
    response = api_client.get("/marketdata/historical-data/")
    assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR


def test_request_specific_data_view_requires_metrics_parameter(api_client):
    response = api_client.get("/marketdata/get-historical-data/")

    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert "metrics' is required" in response.data["error"]


def test_request_specific_data_view_invalid_years_back(api_client):
    response = api_client.get(
        "/marketdata/get-historical-data/", {"metrics": "btc", "years_back": "abc"}
    )
    assert response.status_code == status.HTTP_400_BAD_REQUEST


def test_request_specific_data_view_handles_runtime_error(api_client, mocker):
    mocker.patch(
        "marketdata.services.DataReaderService.DataReaderService.get_market_data",
        side_effect=RuntimeError("Runtime Error"),
    )
    response = api_client.get("/marketdata/get-historical-data/", {"metrics": "btc"})
    assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE


def test_naive_model_view_returns_prediction(api_client, mocker):
    mocker.patch(
        "marketdata.services.ModelService.ModelService.naive", return_value=50000.0
    )
    response = api_client.get("/marketdata/naive-model/")

    assert response.status_code == status.HTTP_200_OK
    assert response.data["close_btc"] == 50000.0


def test_naive_model_view_handles_value_error(api_client, mocker):
    mocker.patch(
        "marketdata.services.ModelService.ModelService.naive",
        side_effect=ValueError("Value Error"),
    )
    response = api_client.get("/marketdata/naive-model/")
    assert response.status_code == status.HTTP_400_BAD_REQUEST


def test_linear_regression_model_view_handles_missing_file(api_client, mocker):
    mocker.patch(
        "marketdata.services.ModelService.ModelService.linear_regression",
        side_effect=FileNotFoundError("Model not found"),
    )
    response = api_client.get("/marketdata/linear-regression-model/")

    assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    assert "Trained linear regression model not found" in response.data["error"]


def test_random_forest_model_view_handles_file_not_found(api_client, mocker):
    mocker.patch(
        "marketdata.services.ModelService.ModelService.random_forest",
        side_effect=FileNotFoundError("Model not found"),
    )
    response = api_client.get("/marketdata/random-forest-model/")
    assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR


def test_lstm_model_view_handles_exception(api_client, mocker):
    mocker.patch(
        "marketdata.services.ModelService.ModelService.lstm_model",
        side_effect=Exception("LSTM Error"),
    )
    response = api_client.get("/marketdata/lstm-model/")
    assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR


def test_tft_model_view_handles_exception(api_client, mocker):
    mocker.patch(
        "marketdata.services.ModelService.ModelService.tft_model",
        side_effect=Exception("TFT Error"),
    )
    response = api_client.get("/marketdata/tft-model/")
    assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
