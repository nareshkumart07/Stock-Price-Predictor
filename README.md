# AI Stock Forecaster

A modular and interview-ready machine learning project that predicts next-day stock values using an LSTM model trained on historical OHLCV data.

## Why This Project Stands Out
- Professional multi-file architecture (separation of concerns).
- Reproducible training (`random_seed` controlled in config).
- Proper time-series split (train/validation/test in chronological order).
- Objective evaluation using MAE, RMSE, MAPE, and a naive baseline.
- Model artifact persistence (`save/load`) for faster demos.
- Yahoo rate-limit handling (retry/backoff + local market-data cache fallback).
- Responsive, user-friendly dashboard with tabbed analytics view.
- Unit tests for core logic and model-store behavior.
- Docker support for portable execution.

## Project Structure

```text
Model/
├── app.py
├── requirements.txt
├── README.md
├── Dockerfile
├── .gitignore
├── artifacts/                      # Auto-generated saved model artifacts
├── screenshots/
│   ├── home.svg
│   ├── forecast.svg
│   └── chart.svg
├── tests/
│   ├── test_chart_service.py
│   ├── test_forecast_service.py
│   ├── test_model_store.py
│   └── test_training_service.py
└── stock_forecaster/
    ├── __init__.py
    ├── config.py
    ├── schemas.py
    ├── model.py
    ├── data_service.py
    ├── training_service.py
    ├── forecast_service.py
    ├── chart_service.py
    ├── model_store.py
    └── ui.py
```

## Screenshots

> Current images are placeholders. Replace them with real screenshots from your running app before final submission.

### Home / Controls Panel
![Home](screenshots/home.svg)

### Forecast Output
![Forecast](screenshots/forecast.svg)

### Price Chart View
![Chart](screenshots/chart.svg)

## Tech Stack
- Python 3.9+
- Streamlit
- PyTorch (LSTM)
- Scikit-learn
- yfinance
- Plotly
- Pytest

## Methodology
1. Fetch historical market data (`Open`, `High`, `Low`, `Close`, `Volume`) from Yahoo Finance.
2. Scale data using `MinMaxScaler`.
3. Build rolling windows (60 timesteps by default).
4. Split time-series data chronologically into train/validation/test.
5. Train the LSTM model on the training split.
6. Evaluate on test split with:
   - MAE (Model vs Baseline)
   - RMSE (Model vs Baseline)
   - MAPE (Model vs Baseline)
   - Directional Accuracy
7. Save trained model + scaler + evaluation summary as reusable artifact.

## Reliability Features
- Avoids fragile `quoteSummary` dependency that frequently triggers Yahoo 429 responses.
- Retries market data download with exponential backoff.
- Uses provider fallback chain: `Yahoo Finance -> Stooq -> local cache`.
- Handles SSL certificate-chain issues in backup provider with controlled fallback strategy.
- Caches ticker market data locally under `artifacts/market_data/`.
- If live fetch fails, app automatically falls back to last cached data.

## Baseline Used
Naive baseline = "next close equals last observed close".

This gives a meaningful benchmark so the LSTM result is measurable, not just visual.

## Reproducibility
- `random_seed` is fixed in [config.py](/Users/fudode/Fudode_Models/Model/stock_forecaster/config.py).
- Artifact path is deterministic per ticker.
- Run configuration and final training/validation losses are stored with artifacts.

## Project Story (For Viva/Interview)
### Problem
Retail users often want a simple educational tool to explore short-term stock movement patterns.

### Approach
I built a Streamlit app using an LSTM forecasting pipeline over OHLCV time-series data, with strict chronological splitting and baseline comparison.

### Result
The app produces next-day OHLCV forecasts, shows benchmark metrics (Model vs Baseline), and reports directional accuracy on held-out test data.

### Limitations
- Single-step forecast only.
- No news/sentiment/macroeconomic signals.
- Financial data remains noisy and non-stationary.

### Future Work
- Multi-step forecasting.
- Feature engineering with technical indicators.
- Hyperparameter optimization and walk-forward validation.

## Local Setup

```bash
cd /Users/fudode/Fudode_Models/Model
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run the App

```bash
python -m streamlit run app.py
```

## Run Tests

```bash
python -m pytest -q
```

## Docker Run

```bash
cd /Users/fudode/Fudode_Models/Model
docker build -t ai-stock-forecaster .
docker run --rm -p 8501:8501 ai-stock-forecaster
```

Then open: `http://localhost:8501`

## Submission Tips (College / Interview)
- Add your name, roll number, and course details at top.
- Replace placeholder screenshots with real app captures.
- Prepare a 2-minute walkthrough: problem, architecture, evaluation, limitations.

## Disclaimer
For educational and demonstration use only.
