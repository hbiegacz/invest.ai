import pytest
import requests
from marketdata.services.BinanceAPIService import BinanceAPIService


def test_binance_service_returns_price_data_on_successful_response(mocker):
    mock_response = mocker.Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"symbol": "BTCUSDT", "price": "50000.00"}
    mocker.patch("requests.get", return_value=mock_response)

    service = BinanceAPIService()
    result = service.get_btc_price()

    assert result == {"symbol": "BTCUSDT", "price": "50000.00"}


def test_binance_service_raises_exception_on_api_error(mocker):
    mock_response = mocker.Mock()
    mock_response.status_code = 500
    mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError()
    mocker.patch("requests.get", return_value=mock_response)

    service = BinanceAPIService()

    with pytest.raises(requests.exceptions.HTTPError):
        service.get_btc_price()
