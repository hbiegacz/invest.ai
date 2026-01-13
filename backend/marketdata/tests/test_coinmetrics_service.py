import pytest
import requests
from marketdata.services.CoinmetricsAPIService import CoinmetricsAPIService


def test_coinmetrics_service_returns_reference_rate_on_success(mocker):
    mock_response = mocker.Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "data": [{"asset": "btc", "ReferenceRateUSD": "50000"}]
    }
    mocker.patch("requests.get", return_value=mock_response)

    service = CoinmetricsAPIService()
    result = service.get_reference_rate()

    assert result == {"data": [{"asset": "btc", "ReferenceRateUSD": "50000"}]}


def test_coinmetrics_service_raises_exception_on_api_failure(mocker):
    mock_response = mocker.Mock()
    mock_response.status_code = 404
    mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError()
    mocker.patch("requests.get", return_value=mock_response)

    service = CoinmetricsAPIService()

    with pytest.raises(requests.exceptions.HTTPError):
        service.get_reference_rate()
