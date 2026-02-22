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


def test_model_service_naive_raises_error_on_missing_column(mocker):
    mock_df = pd.DataFrame({"some_other_column": [100.0]})
    mocker.patch("pandas.read_parquet", return_value=mock_df)
    mocker.patch("os.path.exists", return_value=True)

    service = ModelService()
    with pytest.raises(ValueError, match="Column 'close_btc' not found"):
        service.naive()


def test_model_service_linear_regression_raises_file_not_found(mocker):
    mocker.patch("pathlib.Path.exists", return_value=False)
    service = ModelService()
    with pytest.raises(FileNotFoundError, match="Trained model file not found"):
        service.linear_regression()


def test_model_service_linear_regression_raises_value_error_missing_features(mocker):
    mock_payload = {"model": mocker.Mock()}  # Missing 'features'
    mocker.patch("marketdata.services.ModelService.load", return_value=mock_payload)
    mocker.patch("pathlib.Path.exists", return_value=True)
    mocker.patch("os.path.exists", return_value=True)

    service = ModelService()
    with pytest.raises(ValueError, match="missing 'features' list"):
        service.linear_regression()


def test_model_service_require_columns_raises_on_missing(mocker):
    service = ModelService()
    df = pd.DataFrame({"a": [1]})
    with pytest.raises(ValueError, match="Missing columns in parquet: \['b'\]"):
        service._require_columns(df, ["a", "b"])


def test_model_service_parse_rf_spans_windows():
    service = ModelService()
    features = ["feat_s7", "feat_w14", "other"]
    spans, wins = service._parse_rf_spans_windows(features)
    assert spans == [7]
    assert wins == [14]


def test_model_service_compute_common_features(mocker):
    service = ModelService()
    # Create minimal DF with required columns for _rf_base_columns
    cols = service._rf_base_columns()
    data = {c: [1.0, 2.0] for c in cols}
    data["open_time"] = [pd.Timestamp("2023-01-01"), pd.Timestamp("2023-01-02")]
    df = pd.DataFrame(data)

    processed_df = service._compute_common_features(df, spans=[7], vol_windows=[14])
    assert "hl2_btc" in processed_df.columns
    assert "ret_close_btc" in processed_df.columns
    assert "ewm_ret_close_btc_s7" in processed_df.columns
