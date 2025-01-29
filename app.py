import yfinance as yf
import streamlit as st
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

st.write("""
# NVIDIA Stock Price Prediction

Predict future stock prices using an LSTM model.
""")

tickerSymbol = 'NVDA'
period = st.radio("Select the period", 
                  ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'max'), 
                  index=5, horizontal=True)

data = yf.download(tickerSymbol, period=period)

df = pd.DataFrame(data['Close'])
df.columns = ['Close']

st.write("## Closing Price Data")
col1, col2 = st.columns(2)
with col1:
    st.write(df.head())
with col2:
    st.write(df.tail())

st.write("## Closing Price Chart")
st.line_chart(df)

# Animasi loading dengan spinner
with st.spinner("🔄 Training Model... This may take a while ⏳"):
    progress_bar = st.progress(0)  # Buat progress bar
    for percent_complete in range(1, 101):  
        time.sleep(0.1)  # Simulasi waktu training (bisa disesuaikan)
        progress_bar.progress(percent_complete)  # Update progress
    
    progress_bar.empty()  # Hapus progress bar

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df)

# Function to create dataset
def create_dataset(data, time_step=60):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

time_step = 60
X, y = create_dataset(scaled_data, time_step)
X = X.reshape(X.shape[0], X.shape[1], 1)

train_size = int(len(X) * 0.8)
test_size = len(X) - train_size
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Build and compile the LSTM model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(time_step, 1)),
    LSTM(50, return_sequences=False),
    Dense(25),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, batch_size=32, epochs=10, verbose=1)

# Predictions
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Inverse transform
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)
y_train = scaler.inverse_transform(y_train.reshape(-1, 1))
y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

# Shift predictions
shift_days = 1

train_plot = np.empty_like(scaled_data)
train_plot[:, :] = np.nan
train_plot[time_step + shift_days:time_step + shift_days + len(train_predict), :] = train_predict

test_plot = np.empty_like(scaled_data)
test_plot[:, :] = np.nan
test_start = time_step + len(train_predict) + shift_days + 1
test_end = test_start + len(test_predict)

if test_end > len(test_plot):
    test_end = len(test_plot)

test_plot[test_start:test_end, :] = test_predict[:test_end - test_start]

# **User pilih jumlah hari prediksi**
future_days = st.slider("Select number of days to predict:", min_value=1, max_value=30, value=10)

# **Predict Future Days**
future_predictions = []
future_dates = [df.index[-1] + pd.Timedelta(days=i) for i in range(1, future_days + 1)]

last_60_days = scaled_data[-time_step:].reshape(1, time_step, 1)  # Initial input

for _ in range(future_days):
    next_day_scaled = model.predict(last_60_days)  # Predict next day
    next_day_price = scaler.inverse_transform(next_day_scaled)[0, 0]  # Convert back
    future_predictions.append(next_day_price)

    # Update input with new prediction
    next_input = np.append(last_60_days[0, 1:, 0], next_day_scaled)  # Remove first, add new
    last_60_days = next_input.reshape(1, time_step, 1)

# Extend the DataFrame for plotting
dates = df.index.values[time_step + shift_days:]

combined_df = pd.DataFrame({
    'Date': dates,
    'Actual': scaler.inverse_transform(scaled_data[time_step + shift_days:]).flatten(),
    'Train Predict': train_plot[time_step + shift_days:].flatten(),
    'Test Predict': test_plot[time_step + shift_days:].flatten()
})

# Add Future Predictions
future_df = pd.DataFrame({
    'Date': future_dates,
    'Actual': [np.nan] * future_days,
    'Train Predict': [np.nan] * future_days,
    'Test Predict': [np.nan] * future_days,
    'Future Predict': future_predictions
})

combined_df = pd.concat([combined_df, future_df])
combined_df.set_index('Date', inplace=True)

# Display results
st.write(f"## 🎯 Predicted Closing Prices for the Next {future_days} Days (Latest to Earliest):")
future_df = pd.DataFrame({'Date': future_dates, 'Predicted Price': future_predictions})
# Balik urutan agar yang terbaru muncul duluan
future_df = future_df[::-1].set_index('Date')
st.write(future_df)

# Plot
st.write("### LSTM Predictions vs Actual Prices")
st.line_chart(combined_df)
