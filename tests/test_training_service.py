import numpy as np
import pandas as pd
import pytest

from stock_forecaster.config import ModelConfig
from stock_forecaster.training_service import (
    calculate_regression_metrics,
    create_training_sequences,
    scale_training_data,
    split_time_series_sequences,
)


@pytest.mark.parametrize("row_count, sequence_length", [(25, 5), (8, 5)])
def test_held_out_extremes_do_not_change_scaler_or_training_samples(row_count, sequence_length):
    config = ModelConfig(sequence_length=sequence_length, train_ratio=0.6, val_ratio=0.2)
    data = pd.DataFrame(np.arange(row_count * 5, dtype=float).reshape(row_count, 5))
    raw_X, raw_y = create_training_sequences(data.to_numpy(), sequence_length)
    raw_splits = split_time_series_sequences(raw_X, raw_y, 0.6, 0.2)
    observation_end = sequence_length + len(raw_splits[0])
    changed = data.copy()
    changed.iloc[observation_end:] = 100_000.0
    changed.iloc[-1] = -100_000.0

    scaler, scaled = scale_training_data(data, config)
    changed_scaler, changed_scaled = scale_training_data(changed, config)

    np.testing.assert_array_equal(scaler.data_min_, changed_scaler.data_min_)
    np.testing.assert_array_equal(scaler.data_max_, changed_scaler.data_max_)
    np.testing.assert_array_equal(scaler.data_max_, data.iloc[observation_end - 1])
    np.testing.assert_array_equal(scaled[:observation_end], changed_scaled[:observation_end])
    assert scaler.n_samples_seen_ == observation_end
    assert (changed_scaled[observation_end] > 1).all()
    assert (changed_scaled[-1] < 0).all()

    X, y = create_training_sequences(scaled, sequence_length)
    X_train, y_train, X_val, y_val, X_test, y_test = split_time_series_sequences(X, y, 0.6, 0.2)
    # The first validation target is excluded from fitting while its context is retained.
    np.testing.assert_array_equal(X_val[0], scaled[observation_end - sequence_length:observation_end])
    np.testing.assert_array_equal(y_val[0], scaled[observation_end])
    np.testing.assert_allclose(scaler.inverse_transform(y_test), raw_splits[-1])


def test_training_scaling_rejects_insufficient_history():
    with pytest.raises(ValueError, match="Not enough sequence samples"):
        scale_training_data(pd.DataFrame(np.ones((7, 5))), ModelConfig(sequence_length=5))


def test_create_training_sequences_shape_and_values():
    scaled_data = np.arange(100, dtype=float).reshape(20, 5)

    features, targets = create_training_sequences(scaled_data, sequence_length=5)

    assert features.shape == (15, 5, 5)
    assert targets.shape == (15, 5)
    assert np.array_equal(features[0], scaled_data[0:5])
    assert np.array_equal(targets[0], scaled_data[5])


def test_split_time_series_sequences_chronological_order():
    features = np.arange(300, dtype=float).reshape(20, 3, 5)
    targets = np.arange(100, dtype=float).reshape(20, 5)

    X_train, y_train, X_val, y_val, X_test, y_test = split_time_series_sequences(
        features, targets, train_ratio=0.6, val_ratio=0.2
    )

    assert len(X_train) == 12
    assert len(X_val) == 4
    assert len(X_test) == 4
    assert np.array_equal(X_train[-1], features[11])
    assert np.array_equal(X_val[0], features[12])
    assert np.array_equal(X_test[0], features[16])
    assert np.array_equal(y_test[-1], targets[-1])


def test_calculate_regression_metrics_values():
    y_true = np.array([100.0, 200.0, 300.0])
    y_pred = np.array([110.0, 190.0, 330.0])

    metrics = calculate_regression_metrics(y_true, y_pred)

    assert metrics.mae == pytest.approx(16.666666666666668)
    assert metrics.rmse == pytest.approx(np.sqrt((100 + 100 + 900) / 3))
    assert metrics.mape > 0
