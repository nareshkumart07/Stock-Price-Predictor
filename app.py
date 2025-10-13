import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
from torch.utils.data import TensorDataset, DataLoader
import plotly.graph_objects as go

# --- Page Configuration ---
st.set_page_config(
    page_title="AI Stock Predictor Pro",
    page_icon="💹",
    layout="wide"
)

# --- Model Architecture ---
# The LSTM model definition remains the same.
class StockPriceLSTM(nn.Module):
    def __init__(self, input_size=5, hidden_size=64, num_layers=2, output_size=5, dropout=0.2):
        super(StockPriceLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        return out

# --- Caching Data & Model Functions ---
# Cache data loading to avoid re-downloading on every interaction.
@st.cache_data(ttl="1d")
def load_data(ticker):
    """
    Loads historical stock data for the given ticker.
    Returns the full DataFrame and the last 60 days for prediction.
    """
    try:
        # Fetch 5 years of data to ensure enough history for training
        data = yf.download(ticker, period='5y')[['Open', 'High', 'Low', 'Close', 'Volume']]
        data.dropna(inplace=True)
        if len(data) < 60:
            return None, None, "Insufficient data (less than 60 days)."
        
        last_60_days_df = data.tail(60)
        return data, last_60_days_df, None
    except Exception as e:
        return None, None, f"Error fetching data: {e}"

# Cache the trained model to avoid retraining for the same ticker.
@st.cache_resource
def get_trained_model(ticker, data):
    """
    Preprocesses data and trains the LSTM model for a given ticker.
    Returns the trained model and the scaler object.
    """
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    X, y = [], []
    for i in range(60, len(scaled_data)):
        X.append(scaled_data[i-60:i])
        y.append(scaled_data[i])
    X, y = np.array(X), np.array(y)

    model = StockPriceLSTM()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    epochs = 20  # Increased epochs for potentially better accuracy
    model.train()
    for epoch in range(epochs):
        for batch_X, batch_y in loader:
            output = model(batch_X)
            loss = criterion(output, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model, scaler

# --- UI and Main Application Logic ---
st.title("💹 AI Stock Predictor Pro")
st.markdown("An advanced tool using LSTM neural networks to forecast the next day's stock values.")

# --- Sidebar for User Inputs ---
with st.sidebar:
    st.header("Control Panel")
    ticker_input = st.text_input("Enter Stock Ticker", "AAPL").upper()
    predict_button = st.button("Predict Next Day", type="primary")
    st.markdown("---")
    st.info("Disclaimer: This is a demo for educational purposes and should not be used for financial decisions.")

# --- Main Content Area ---
if predict_button:
    if not ticker_input:
        st.warning("Please enter a stock ticker.")
    else:
        with st.spinner(f"Analyzing {ticker_input}... This may take a moment."):
            # 1. Load Data
            stock_data, last_60_days, error = load_data(ticker_input)

            if error:
                st.error(error)
            else:
                # 2. Get Company Info
                try:
                    info = yf.Ticker(ticker_input).info
                    company_name = info.get('longName', 'N/A')
                    sector = info.get('sector', 'N/A')
                    logo_url = info.get('logo_url', '')
                except Exception:
                    company_name, sector, logo_url = 'N/A', 'N/A', ''

                # 3. Train Model
                model, scaler = get_trained_model(ticker_input, stock_data)

                # 4. Make Prediction
                last_60_days_scaled = scaler.transform(last_60_days)
                input_tensor = torch.tensor(last_60_days_scaled, dtype=torch.float32).unsqueeze(0)
                
                model.eval()
                with torch.no_grad():
                    predicted_scaled = model(input_tensor)
                    prediction_original = scaler.inverse_transform(predicted_scaled.numpy())

                prediction = prediction_original[0]
                last_close = stock_data['Close'].iloc[-1]
                trend = "Up" if prediction[3] > last_close else "Down"
                trend_icon = "🔼" if trend == "Up" else "🔽"

                # --- Display Results ---
                st.header(f"Analysis for {company_name} ({ticker_input})")
                
                # Display Company Info & Prediction Metrics
                col1, col2, col3 = st.columns([1, 4, 2])
                with col1:
                    if logo_url:
                        st.image(logo_url, width=100)
                with col2:
                    st.metric(label="**Predicted Close Price**", value=f"${prediction[3]:,.2f}", delta=f"{prediction[3] - last_close:,.2f} ({trend}) {trend_icon}")
                    st.write(f"**Sector:** {sector}")
                
                st.subheader("Next Day's Predicted Values")
                cols = st.columns(4)
                cols[0].metric("Open", f"${prediction[0]:,.2f}")
                cols[1].metric("High", f"${prediction[1]:,.2f}")
                cols[2].metric("Low", f"${prediction[2]:,.2f}")
                cols[3].metric("Volume", f"{prediction[4]:,.0f}")
                
                # Display Interactive Chart
                st.subheader("Historical Price Chart (Last 5 Years)")
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], mode='lines', name='Close Price'))
                fig.update_layout(
                    title=f'{ticker_input} Closing Price History',
                    xaxis_title='Date',
                    yaxis_title='Price (USD)',
                    xaxis_rangeslider_visible=True,
                    template='plotly_dark'
                )
                st.plotly_chart(fig, use_container_width=True)

                # Display Recent Data Table
                with st.expander("View Recent Historical Data"):
                    st.dataframe(stock_data.tail(10).style.format("{:,.2f}"))

else:
    st.info("Enter a stock ticker in the sidebar and click 'Predict Next Day' to begin.")
