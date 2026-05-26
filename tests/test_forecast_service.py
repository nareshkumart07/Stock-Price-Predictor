import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.preprocessing import MinMaxScaler

from stock_forecaster.forecast_service import predict_next_day


class DummyModel:
    def eval(self):
        return self

    def __call__(self, _x):
        return torch.tensor([[0.6, 0.7, 0.5, 0.8, 0.4]], dtype=torch.float32)


def test_predict_next_day_returns_expected_values():
    data = pd.DataFrame(
        {
            "Open": np.linspace(100, 200, 80),
            "High": np.linspace(110, 210, 80),
            "Low": np.linspace(90, 190, 80),
            "Close": np.linspace(105, 205, 80),
            "Volume": np.linspace(1000, 5000, 80),
        }
    )

    scaler = MinMaxScaler().fit(data)
    expected = scaler.inverse_transform(np.array([[0.6, 0.7, 0.5, 0.8, 0.4]]))[0]

    result = predict_next_day(
        model=DummyModel(),
        scaler=scaler,
        last_window=data.tail(60),
        last_close=float(data["Close"].iloc[-1]),
    )

    assert result.open_price == pytest.approx(expected[0], rel=1e-5)
    assert result.high_price == pytest.approx(expected[1], rel=1e-5)
    assert result.low_price == pytest.approx(expected[2], rel=1e-5)
    assert result.close_price == pytest.approx(expected[3], rel=1e-5)
    assert result.volume == pytest.approx(expected[4], rel=1e-5)
    assert result.trend == "Down"
