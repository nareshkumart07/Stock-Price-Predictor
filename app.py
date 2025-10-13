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
    page_title="AI Stock Forecaster",
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
# Cache data and info loading to avoid re-downloading on every interaction.
@st.cache_data(ttl="1d")
def load_data_and_info(ticker):
    """
    Loads historical stock data and company info for the given ticker.
    Returns dataframes, a dictionary of company info, and an error message if applicable.
    """
    try:
        ticker_obj = yf.Ticker(ticker)
        
        # Fetch company info
        info = ticker_obj.info
        if not info or info.get('regularMarketPrice') is None: # Check if the ticker is valid
             return None, None, None, f"Could not find the stock symbol: {ticker}. Please check if it's correct."

        company_profile = {
            "name": info.get('longName', 'N/A'),
            "sector": info.get('sector', 'N/A'),
            "logo": info.get('logo_url', '')
        }
        
        # Fetch 5 years of data using the Ticker object
        data = ticker_obj.history(period='5y')[['Open', 'High', 'Low', 'Close', 'Volume']]
        data.dropna(inplace=True)
        
        if len(data) < 60:
            return None, None, None, f"Not enough past information for {ticker} to make a good guess."
        
        last_60_days_df = data.tail(60)
        
        return data, last_60_days_df, company_profile, None
    except Exception as e:
        return None, None, None, f"Something went wrong when trying to get information for {ticker}."

# Cache the trained model to avoid retraining for the same ticker.
@st.cache_resource
def get_trained_model(ticker, data):
    """
    Preprocesses data and trains the AI model.
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

    epochs = 20
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
st.title("💹 AI Stock Price Forecaster")
st.markdown("This tool uses Artificial Intelligence to guess the next day's stock prices.")

# --- Sidebar for User Inputs ---
with st.sidebar:
    st.header("Settings")
    ticker_input = st.text_input("Enter a Company's Stock Symbol", "AAPL").upper()
    predict_button = st.button("Forecast Next Day's Price", type="primary")
    st.markdown("---")
    st.info("Disclaimer: This is just for fun and learning. Please don't use it to make real investment choices.")

# --- Main Content Area ---
if predict_button:
    if not ticker_input:
        st.warning("Please enter a stock symbol.")
    else:
        with st.spinner(f"Thinking about {ticker_input}... This might take a moment."):
            try:
                # 1. Load Data and Info
                stock_data, last_60_days, company_profile, error = load_data_and_info(ticker_input)

                if error:
                    st.error(error)
                else:
                    # 2. Train Model
                    model, scaler = get_trained_model(ticker_input, stock_data)

                    # 3. Make Prediction
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
                    st.header(f"Forecast for {company_profile['name']} ({ticker_input})")
                    
                    col1, col2 = st.columns([1, 4])
                    with col1:
                        if company_profile['logo']:
                            st.image(company_profile['logo'], width=100)
                    with col2:
                        st.metric(label="**Estimated Closing Price**", value=f"${prediction[3]:,.2f}", delta=f"{prediction[3] - last_close:,.2f} ({trend}) {trend_icon}")
                        st.write(f"**Business Sector:** {company_profile['sector']}")
                    
                    st.subheader("Tomorrow's Price Estimates")
                    cols = st.columns(4)
                    cols[0].metric("Open", f"${prediction[0]:,.2f}")
                    cols[1].metric("High", f"${prediction[1]:,.2f}")
                    cols[2].metric("Low", f"${prediction[2]:,.2f}")
                    cols[3].metric("Volume", f"{prediction[4]:,.0f}")
                    
                    # Display Interactive Chart
                    st.subheader("Price History (Last 5 Years)")
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], mode='lines', name='Close Price'))
                    fig.update_layout(
                        title=f"How {ticker_input}'s Price Changed Over Time",
                        xaxis_title='Date',
                        yaxis_title='Price (USD)',
                        xaxis_rangeslider_visible=True,
                        template='plotly_dark'
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Display Recent Data Table
                    with st.expander("See the Last 10 Days of Data"):
                        st.dataframe(stock_data.tail(10).style.format("{:,.2f}"))

            except Exception as e:
                st.error("Sorry, something unexpected happened. Please try again.")
else:
    st.info("Enter a company's stock symbol in the settings panel on the left and click 'Forecast Next Day's Price' to start.")

