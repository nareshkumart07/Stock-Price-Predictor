# AI Stock Forecaster

An educational Streamlit application for exploring next-day stock forecasts using a PyTorch LSTM trained on historical OHLCV data. Evaluation compares the model with a last-observed-close baseline; forecasts are not investment advice.

## Features
- Professional multi-file architecture (separation of concerns).
- Reproducible training (`random_seed` controlled in config).
- Chronological train/validation/test targets, with preprocessing fitted only on training observations.
- Objective evaluation using MAE, RMSE, MAPE, and a naive baseline.
- Model artifact persistence (`save/load`) for faster demos.
- Yahoo rate-limit handling (retry/backoff + local market-data cache fallback).
- Responsive, user-friendly dashboard with tabbed analytics view.
- Unit tests for core logic and model-store behavior.
- Docker support for portable execution.

## Project Structure

```text
Stock-Price-Predictor/
├── app.py
├── requirements.txt
├── README.md
├── Dockerfile
├── .gitignore
├── artifacts/                      # Auto-generated saved model artifacts
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
2. Define chronological training, validation, and test boundaries over prediction targets.
3. Fit `MinMaxScaler` only on training-period observations, including the last training target, then transform the complete history without refitting or clipping.
4. Build rolling windows (60 timesteps by default) and split them at those boundaries. Held-out windows can use earlier observed prices as context; future targets do not affect preprocessing.
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
- `random_seed` is fixed in [config.py](stock_forecaster/config.py).
- Artifact path is deterministic per ticker.
- Run configuration and final training/validation losses are stored with artifacts.
- Legacy artifacts created before train-only scaling are rejected and retrained automatically.
- Historical accuracy is not a promise of future performance. Report the ticker, date range, split sizes, and model-versus-baseline metrics when sharing an evaluation.

## Project Background
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
git clone https://github.com/nareshkumart07/Stock-Price-Predictor.git
cd Stock-Price-Predictor
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
# From the repository root
docker build -t ai-stock-forecaster .
docker run --rm -p 8501:8501 ai-stock-forecaster
```

Then open: `http://localhost:8501`

## Disclaimer
For educational and demonstration use only.
