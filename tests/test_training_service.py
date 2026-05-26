import numpy as np
import pytest

from stock_forecaster.training_service import (
    calculate_regression_metrics,
    create_training_sequences,
    split_time_series_sequences,
)


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
