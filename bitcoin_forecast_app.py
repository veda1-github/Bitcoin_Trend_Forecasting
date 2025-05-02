import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(layout="wide")

# Step 1: Load Data
st.title("📈 Bitcoin Price Prediction with LSTM")
if st.button("🔄 Load Bitcoin Data"):
    df = yf.download("BTC-USD", start="2020-01-01", end="2024-12-31", interval="1d")
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
    st.success("Data Loaded!")
    st.line_chart(df['Close'])

# Step 2: Show Volume
if st.button("📊 Show Volume"):
    df = yf.download("BTC-USD", start="2020-01-01", end="2024-12-31", interval="1d")
    st.line_chart(df['Volume'])

# Step 3: Preprocess & Train
if st.button("🧠 Train LSTM Model"):
    df = yf.download("BTC-USD", start="2020-01-01", end="2024-12-31", interval="1d")
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
    scaler = MinMaxScaler()
    close_scaled = scaler.fit_transform(df[['Close']])

    sequence_length = 60
    X, y = [], []
    for i in range(len(close_scaled) - sequence_length):
        X.append(close_scaled[i:i+sequence_length])
        y.append(close_scaled[i+sequence_length])
    X, y = np.array(X), np.array(y)

    train_size = int(len(X) * 0.8)
    X_train, y_train = X[:train_size], y[:train_size]

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=64, shuffle=True)

    class LSTMModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(1, 50, 2, batch_first=True)
            self.fc = nn.Linear(50, 1)

        def forward(self, x):
            h0 = torch.zeros(2, x.size(0), 50)
            c0 = torch.zeros(2, x.size(0), 50)
            out, _ = self.lstm(x, (h0, c0))
            return self.fc(out[:, -1, :])

    model = LSTMModel()
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(5):
        for xb, yb in train_loader:
            preds = model(xb)
            loss = loss_fn(preds, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        st.write(f"Epoch {epoch+1}, Loss: {loss.item():.6f}")
    st.success("Model Trained!")

    st.session_state.model = model
    st.session_state.scaler = scaler
    st.session_state.X_test = X[train_size:]
    st.session_state.y_test = y[train_size:]

# Step 4: Predict
if st.button("🔮 Predict & Show Chart"):
    model = st.session_state.model
    scaler = st.session_state.scaler
    X_test = st.session_state.X_test
    y_test = st.session_state.y_test

    model.eval()
    with torch.no_grad():
        test_preds = model(torch.tensor(X_test, dtype=torch.float32)).numpy()
    
    predicted_prices = scaler.inverse_transform(test_preds)
    actual_prices = scaler.inverse_transform(y_test)

    st.line_chart(pd.DataFrame({
        'Actual': actual_prices.flatten(),
        'Predicted': predicted_prices.flatten()
    }))

# Step 5: Forecast
if st.button("📅 Forecast Next 30 Days"):
    model = st.session_state.model
    scaler = st.session_state.scaler
    last_input = st.session_state.X_test[-1]

    future_preds = []
    for _ in range(30):
        with torch.no_grad():
            next_pred = model(torch.tensor(last_input[np.newaxis, :, :], dtype=torch.float32)).numpy()
        future_preds.append(next_pred[0, 0])
        last_input = np.append(last_input[1:], [[next_pred[0, 0]]], axis=0)

    future_prices = scaler.inverse_transform(np.array(future_preds).reshape(-1, 1))
    st.line_chart(pd.DataFrame(future_prices, columns=["Forecast Price"]))
