import pytest
import requests
from marketdata.services.StooqAPIService import StooqAPIService


def test_stooq_service_returns_parsed_csv_data_on_success(mocker):
    csv_content = "Symbol,Date,Time,Open,High,Low,Close,Volume\n^spx,2024-01-01,10:00,5000,5010,4990,5005,1000000"
    mock_response = mocker.Mock()
    mock_response.status_code = 200
    mock_response.text = csv_content
    mocker.patch("requests.get", return_value=mock_response)

    service = StooqAPIService()
    result = service.get_sp500_data()

    assert len(result) == 1
    assert result[0]["Symbol"] == "^spx"
    assert result[0]["Close"] == "5005"


def test_stooq_service_raises_exception_on_api_error(mocker):
    mock_response = mocker.Mock()
    mock_response.status_code = 503
    mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError()
    mocker.patch("requests.get", return_value=mock_response)

    service = StooqAPIService()

    with pytest.raises(requests.exceptions.HTTPError):
        service.get_sp500_data()
