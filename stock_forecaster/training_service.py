import random
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

from stock_forecaster.config import CLOSE_COLUMN, FEATURE_COLUMNS, ModelConfig
from stock_forecaster.model import StockPriceLSTM
from stock_forecaster.schemas import EvaluationSummary, RegressionMetrics


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def create_training_sequences(
    scaled_data: np.ndarray, sequence_length: int
) -> Tuple[np.ndarray, np.ndarray]:
    features, targets = [], []

    for i in range(sequence_length, len(scaled_data)):
        features.append(scaled_data[i - sequence_length : i])
        targets.append(scaled_data[i])

    return np.array(features), np.array(targets)


def split_time_series_sequences(
    features: np.ndarray, targets: np.ndarray, train_ratio: float, val_ratio: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_end, val_end = _time_series_split_boundaries(
        len(features), train_ratio, val_ratio
    )

    X_train, y_train = features[:train_end], targets[:train_end]
    X_val, y_val = features[train_end:val_end], targets[train_end:val_end]
    X_test, y_test = features[val_end:], targets[val_end:]

    return X_train, y_train, X_val, y_val, X_test, y_test


def _time_series_split_boundaries(
    total: int, train_ratio: float, val_ratio: float
) -> Tuple[int, int]:
    if not 0 < train_ratio < 1 or not 0 < val_ratio < 1 or train_ratio + val_ratio >= 1:
        raise ValueError("Train and validation ratios must leave a non-empty test split.")
    if total < 3:
        raise ValueError("Not enough sequence samples for train/validation/test split.")

    train_end = max(1, int(total * train_ratio))
    val_end = max(train_end + 1, int(total * (train_ratio + val_ratio)))

    if val_end >= total:
        val_end = total - 1
    if train_end >= val_end:
        train_end = max(1, val_end - 1)

    return train_end, val_end


def scale_training_data(
    data: pd.DataFrame, config: ModelConfig
) -> Tuple[MinMaxScaler, np.ndarray]:
    """Fit on training observations only, including the last training target.

    Validation/test windows may use earlier observations as context, but their
    targets never influence fitted preprocessing. Held-out values can exceed
    [0, 1]; clipping them would hide genuine changes in the market distribution.
    """
    if config.sequence_length < 1:
        raise ValueError("Sequence length must be positive.")
    train_end, _ = _time_series_split_boundaries(
        len(data) - config.sequence_length, config.train_ratio, config.val_ratio
    )
    train_observation_end = config.sequence_length + train_end
    scaler = MinMaxScaler()
    scaler.fit(data.iloc[:train_observation_end])
    return scaler, scaler.transform(data)


def build_model(config: ModelConfig) -> StockPriceLSTM:
    return StockPriceLSTM(
        input_size=config.input_size,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        output_size=config.output_size,
        dropout=config.dropout,
    )


def inverse_transform_feature(
    scaled_values: np.ndarray, scaler: MinMaxScaler, feature_index: int
) -> np.ndarray:
    return (scaled_values - scaler.min_[feature_index]) / scaler.scale_[feature_index]


def calculate_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> RegressionMetrics:
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    denominator = np.maximum(np.abs(y_true), 1e-8)
    mape = float(np.mean(np.abs((y_true - y_pred) / denominator)) * 100.0)
    return RegressionMetrics(mae=mae, rmse=rmse, mape=mape)


def evaluate_close_price_predictions(
    model: StockPriceLSTM,
    scaler: MinMaxScaler,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Tuple[RegressionMetrics, RegressionMetrics, float]:
    close_index = FEATURE_COLUMNS.index(CLOSE_COLUMN)

    model.eval()
    with torch.no_grad():
        preds_scaled = model(torch.tensor(X_test, dtype=torch.float32)).cpu().numpy()

    y_true_close = inverse_transform_feature(y_test[:, close_index], scaler, close_index)
    y_pred_close = inverse_transform_feature(preds_scaled[:, close_index], scaler, close_index)

    baseline_scaled = X_test[:, -1, close_index]
    baseline_close = inverse_transform_feature(baseline_scaled, scaler, close_index)

    model_metrics = calculate_regression_metrics(y_true_close, y_pred_close)
    baseline_metrics = calculate_regression_metrics(y_true_close, baseline_close)

    true_change = np.sign(y_true_close - baseline_close)
    pred_change = np.sign(y_pred_close - baseline_close)
    direction_accuracy = float(np.mean(true_change == pred_change) * 100.0)

    return model_metrics, baseline_metrics, direction_accuracy


def train_model(
    data: pd.DataFrame, config: ModelConfig
) -> Tuple[StockPriceLSTM, MinMaxScaler, EvaluationSummary, Dict[str, float]]:
    set_random_seed(config.random_seed)

    scaler, scaled_data = scale_training_data(data, config)

    X, y = create_training_sequences(scaled_data, config.sequence_length)
    X_train, y_train, X_val, y_val, X_test, y_test = split_time_series_sequences(
        X, y, train_ratio=config.train_ratio, val_ratio=config.val_ratio
    )

    model = build_model(config)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=False)

    final_train_loss = 0.0
    final_val_loss = 0.0

    model.train()
    for _ in range(config.epochs):
        batch_losses = []
        for batch_X, batch_y in train_loader:
            output = model(batch_X)
            loss = criterion(output, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())

        final_train_loss = float(np.mean(batch_losses)) if batch_losses else 0.0

        with torch.no_grad():
            val_output = model(torch.tensor(X_val, dtype=torch.float32))
            val_loss = criterion(val_output, torch.tensor(y_val, dtype=torch.float32))
            final_val_loss = float(val_loss.item())

    model_metrics, baseline_metrics, direction_accuracy = evaluate_close_price_predictions(
        model=model,
        scaler=scaler,
        X_test=X_test,
        y_test=y_test,
    )

    evaluation = EvaluationSummary(
        model_metrics=model_metrics,
        baseline_metrics=baseline_metrics,
        direction_accuracy=direction_accuracy,
        train_size=len(X_train),
        val_size=len(X_val),
        test_size=len(X_test),
    )

    run_config = {
        "epochs": float(config.epochs),
        "batch_size": float(config.batch_size),
        "learning_rate": float(config.learning_rate),
        "sequence_length": float(config.sequence_length),
        "seed": float(config.random_seed),
        "final_train_loss": final_train_loss,
        "final_val_loss": final_val_loss,
    }

    return model, scaler, evaluation, run_config
