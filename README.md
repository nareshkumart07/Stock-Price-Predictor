# 📈 Stock Price Predictor (NSE) using LSTM/GRU

This project is a deep learning-based stock price prediction system that forecasts **next-day Open, High, Low, Close, and Volume** for any **NSE-listed stock** using historical data from Yahoo Finance.

It uses **LSTM neural networks** to analyze the past 60 days of stock data and generate accurate predictions for the following day.

---

## 🚀 Features

- Predicts **Open, High, Low, Close, Volume**
- Supports **any NSE stock symbol** (e.g., `RELIANCE.NS`, `TCS.NS`)
- Uses **last 60 days** of historical data
- Built using **PyTorch**, **yfinance**, and **scikit-learn**
- Clean, modular code for training and inference
- Optional **Streamlit UI** for user interaction

---

## 🧠 Model Architecture

- LSTM layers
- Two Fully Connected (Dense) layers
- ReLU activations
- MSE Loss + Adam Optimizer

---

## 📦 Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
