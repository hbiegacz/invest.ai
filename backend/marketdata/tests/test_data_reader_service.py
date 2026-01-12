import pytest
import os
import pandas as pd
from datetime import datetime, timedelta
from marketdata.services.DataReaderService import DataReaderService

def test_data_reader_service_is_data_old_returns_true_if_file_missing(mocker):
    mocker.patch("os.path.exists", return_value=False)
    service = DataReaderService()
    assert service._is_data_old() is True

def test_data_reader_service_is_data_old_returns_true_if_file_outdated(mocker):
    mocker.patch("os.path.exists", return_value=True)
    outdated_time = datetime.now() - timedelta(hours=25)
    mocker.patch("os.path.getmtime", return_value=outdated_time.timestamp())
    service = DataReaderService()
    assert service._is_data_old() is True

def test_data_reader_service_is_data_old_returns_false_if_file_fresh(mocker):
    mocker.patch("os.path.exists", return_value=True)
    fresh_time = datetime.now() - timedelta(hours=10)
    mocker.patch("os.path.getmtime", return_value=fresh_time.timestamp())
    service = DataReaderService()
    assert service._is_data_old() is False

def test_data_reader_service_get_market_data_refreshes_if_needed(mocker):
    service = DataReaderService()
    mocker.patch.object(service, "_is_data_old", return_value=True)
    mock_refresh = mocker.patch.object(service, "_refresh_data_file")
    
    mock_df = pd.DataFrame({"open_time": [pd.Timestamp.now()], "close_btc": [50000]})
    mocker.patch.object(service, "_load_filtered_frame", return_value=mock_df)
    
    service.get_market_data(requested_metrics=["close_btc"])
    
    mock_refresh.assert_called_once()

def test_data_reader_service_load_filtered_frame_filters_by_date(mocker, tmp_path):
    df = pd.DataFrame({
        "open_time": [
            pd.Timestamp.now() - pd.DateOffset(years=11),
            pd.Timestamp.now() - pd.DateOffset(years=5)
        ],
        "close_btc": [30000, 50000]
    })
    
    service = DataReaderService()
    mocker.patch("pandas.read_parquet", return_value=df)
    mocker.patch.object(service, "_is_data_insufficient", return_value=False)
    
    result_df = service._load_filtered_frame(["close_btc", "open_time"], years_back=10)
    
    assert len(result_df) == 1
    assert result_df.iloc[0]["close_btc"] == 50000
