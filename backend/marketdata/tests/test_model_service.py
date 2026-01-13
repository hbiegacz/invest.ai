import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from marketdata.services.ModelService import ModelService


def test_model_service_naive_returns_last_close(mocker):
    mock_df = pd.DataFrame({"close_btc": [100.0, 200.0, 150.0]})
    mocker.patch("pandas.read_parquet", return_value=mock_df)
    mocker.patch("os.path.exists", return_value=True)

    service = ModelService()
    result = service.naive()

    assert result == 150.0


def test_model_service_naive_raises_error_on_empty_data(mocker):
    mocker.patch("pandas.read_parquet", return_value=pd.DataFrame())
    mocker.patch("os.path.exists", return_value=True)

    service = ModelService()
    with pytest.raises(ValueError, match="No data available"):
        service.naive()


def test_model_service_linear_regression_calls_predict_with_last_row(mocker):
    mock_payload = {"model": mocker.Mock(), "features": ["feat1", "feat2"]}
    mock_pipeline = mock_payload["model"]
    mock_scaler = mocker.Mock()
    mock_scaler.feature_names_in_ = ["feat1", "feat2"]
    mock_pipeline.named_steps.get.return_value = mock_scaler
    mock_pipeline.predict.return_value = [0.1]

    mocker.patch("marketdata.services.ModelService.load", return_value=mock_payload)
    mocker.patch("pathlib.Path.exists", return_value=True)
    mocker.patch("os.path.exists", return_value=True)

    mock_df = pd.DataFrame(
        {
            "open_time": [1, 2],
            "feat1": [1.0, 3.0],
            "feat2": [2.0, 4.0],
            "close_btc": [1000.0, 2000.0],
        }
    )
    mocker.patch("pandas.read_parquet", return_value=mock_df)

    service = ModelService()
    service.linear_regression()

    called_args = mock_pipeline.predict.call_args[0][0]
    assert np.array_equal(called_args, np.array([[3.0, 4.0]]))


def test_model_service_rf_base_columns_contains_required_fields():
    service = ModelService()
    cols = service._rf_base_columns()
    assert "open_time" in cols
    assert "gdp" in cols
    assert "close_btc" in cols
    assert "volume_spx" in cols
