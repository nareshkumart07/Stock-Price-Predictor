import torch
from sklearn.preprocessing import MinMaxScaler

from stock_forecaster.schemas import ForecastResult


def predict_next_day(model, scaler: MinMaxScaler, last_window, last_close: float) -> ForecastResult:
    model.eval()
    last_window_scaled = scaler.transform(last_window)
    input_tensor = torch.tensor(last_window_scaled, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        predicted_scaled = model(input_tensor)
        prediction_original = scaler.inverse_transform(predicted_scaled.numpy())[0]

    close_price = float(prediction_original[3])
    delta_close = close_price - last_close
    trend = "Up" if close_price > last_close else "Down"

    return ForecastResult(
        open_price=float(prediction_original[0]),
        high_price=float(prediction_original[1]),
        low_price=float(prediction_original[2]),
        close_price=close_price,
        volume=float(prediction_original[4]),
        delta_close=delta_close,
        trend=trend,
    )
