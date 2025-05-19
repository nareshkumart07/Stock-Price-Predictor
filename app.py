# deploy it using streamlit where user can get any stock prediction 

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import datetime
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
from torch.utils.data import TensorDataset, DataLoader

# Define the LSTM Architecture (same as before)
class StockPriceLSTM(nn.Module):
    def __init__(self, input_size=5, hidden_size=64, num_layers=2, output_size=5, dropout=0.2):
        super(StockPriceLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)

        self.fc1 = nn.Linear(hidden_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0,c0))
        out = out[:, -1, :]

        out = self.relu(self.fc1(out))
        out = self.relu(self.fc2(out))
        out = self.fc3(out)

        return out

# Function to get data and scale it

def load_and_preprocess_data(ticker, period='1y'):
    data = yf.download(ticker, period=period)[['Open', 'High', 'Low', 'Close', 'Volume']]
    data.dropna(inplace=True)

    if len(data) < 60:
        st.error("Insufficient data to make a prediction. Please try a different ticker or period.")
        return None, None, None

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    X, y = [], []
    for i in range(60, len(scaled_data)):
        X.append(scaled_data[i-60:i])
        y.append(scaled_data[i])

    X, y = np.array(X), np.array(y)

    return X, y, scaler

# Function to train the model

def train_model(X, y):
    model = StockPriceLSTM()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    epochs = 10 # Reduced epochs for faster demonstration
    model.train()

    progress_bar = st.progress(0)
    status_text = st.empty()

    for epoch in range(epochs):
        for i, (batch_X, batch_y) in enumerate(loader):
            output = model(batch_X)
            loss = criterion(output, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        status_text.text(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
        progress_bar.progress((epoch + 1) / epochs)

    return model

# Streamlit App

st.title("Stock Price Predictor (LSTM)")
# style the title
st.markdown(
    """
    <style>
    .title {
        font-size: 40px;
        color: #4CAF50;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)
# Add a description and instructions for the user
st.markdown(
    """
    <style>
    .description {
        font-size: 20px;
        color: #555;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.write("Enter a stock ticker symbol to predict the next day's Open, High, Low, Close, and Volume.")
st.write("This model uses historical stock data to predict future prices using an LSTM neural network.")
st.write("Note: The model is trained on the fly, so it may take a few moments to get predictions.")
st.write("Please ensure you have a stable internet connection for fetching stock data.")
st.write("Disclaimer: This is a demo model and should not be used for actual trading decisions.")
# execption handling for ticker input
try:
    # Input for stock ticker
    # This will be the ticker symbol for the stock
    
    ticker_input = st.text_input("Stock Ticker (e.g., AAPL)", "AAPL").upper()
except Exception as e:
    st.error(f"Error in ticker input: {e}")
    ticker_input = None

# Button to trigger prediction
if st.button("Predict Next Day's Prices"):
    # Check if ticker_input is not empty
    if ticker_input:
        st.write(f"Fetching data for {ticker_input}...")
        X, y, scaler = load_and_preprocess_data(ticker_input)

        if X is not None and y is not None and scaler is not None:
            st.write("Data loaded and preprocessed successfully.")

            # Display recent data
            st.subheader("Recent Stock Data")
            # Fetch the last few days of data to show
            recent_data = yf.download(ticker_input, period='10d')[['Open', 'High', 'Low', 'Close', 'Volume']]
            st.dataframe(recent_data)

            st.write("Training the LSTM model...")
            model = train_model(X, y)
            st.write("Model training complete.")

            st.subheader("Predicted Next Day's Values:")
            model.eval()
            with torch.no_grad():
                # Take the last 60 days from the original scaled data
                last_60_days = scaler.transform(yf.download(ticker_input, period='70d')[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()[-60:]) # Fetch a bit more to ensure 60 days
                input_tensor = torch.tensor(last_60_days, dtype=torch.float32).unsqueeze(0)

                # Make prediction
                predicted_scaled_data = model(input_tensor)

                # Inverse transform
                predicted_original_data = scaler.inverse_transform(predicted_scaled_data.numpy())

            prediction = predicted_original_data[0]
            # Display the prediction in Bold
            st.markdown(f"**Open:** {prediction[0]:,.2f}  \n"
                        f"**High:** {prediction[1]:,.2f}  \n"
                        f"**Low:** {prediction[2]:,.2f}  \n"
                        f"**Close:** {prediction[3]:,.2f}  \n"
                        f"**Volume:** {prediction[4]:,.0f}")
            
            st.success("Prediction completed successfully!")
            st.write("Note: The prediction is based on the last 60 days of data and may not reflect real market conditions.")

        else:
            st.warning("Could not process data for the given ticker. Please ensure it's a valid ticker and sufficient data is available.")
