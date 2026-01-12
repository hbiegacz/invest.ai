import pytest
import time
import pandas as pd
import requests
import io
import os
from pathlib import Path
from marketdata.services.HistoricalDataService import HistoricalDataService

def test_historical_data_service_apply_exclusions_removes_specified_columns():
    df = pd.DataFrame({
        "open_time": [1, 2],
        "open_btc": [10, 20],
        "close_btc": [11, 21],
        "volume_btc": [100, 200]
    })
    service = HistoricalDataService()
    
    result_df = service._apply_exclusions(df, excluded_cols=["open_btc", "volume_btc"])
    
    assert "open_btc" not in result_df.columns
    assert "volume_btc" not in result_df.columns
    assert "close_btc" in result_df.columns
    assert "open_time" in result_df.columns

def test_historical_data_service_fetch_spx_df_parses_csv_correctly(mocker):
    csv_content = "Data,Otwarcie,Najwyzszy,Najnizszy,Zamkniecie,Wolumen\n2024-01-01,5000,5010,4990,5005,1000000"
    mock_resp = mocker.Mock()
    mock_resp.status_code = 200
    mock_resp.text = csv_content
    mocker.patch("requests.get", return_value=mock_resp)
    
    service = HistoricalDataService()
    df = service.fetch_spx_df()
    
    assert not df.empty
    assert "open_time" in df.columns
    assert "close_spx" in df.columns
    assert df.iloc[0]["close_spx"] == 5005

def test_historical_data_service_fetch_historical_df_handles_binance_batches(mocker):
    mock_binance_data = [[int(time.time() * 1000), "10", "11", "9", "10.5", "100", int(time.time() * 1000) + 1000, "1000", 10, "5", "50", "0"]]
    
    def mock_fetch(url):
        return mock_binance_data
        
    service = HistoricalDataService(years_back=1)
    mocker.patch.object(service, "_fetch", side_effect=mock_fetch)
    
    df = service.fetch_historical_df("BTCUSDC", years_back=1)
    
    assert not df.empty
    assert "open_btc" not in df.columns
    assert "close" in df.columns 
    assert len(df) >= 1

def test_historical_data_service_generate_parquet_file_saves_to_disk(mocker, tmp_path):
    mocker.patch("django.conf.settings.BASE_DIR", str(tmp_path))
    
    mock_df = pd.DataFrame({
        "open_time": pd.to_datetime(["2024-01-01"]),
        "close_btc": [50000.0]
    })
    
    service = HistoricalDataService()
    mocker.patch.object(service, "fetch_multi_symbol_df", return_value=mock_df)
    
    filename = "test_data.parquet"
    result = service.generate_parquet_file(filename=filename)
    
    expected_path = Path(tmp_path) / "data" / filename
    assert expected_path.exists()
    assert result["file"] == str(expected_path)
