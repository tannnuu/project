import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import streamlit as st

# Load the trained model
model = load_model('/Users/abdullah/Desktop/project/Bitcoin_Price_prediction_Model.keras')

# Streamlit app header
st.header('Bitcoin Price Prediction Model')
st.subheader('Bitcoin Price Data')

# Download Bitcoin data using yfinance
data = pd.DataFrame(yf.download('BTC-USD', '2015-01-01', '2024-10-30'))
data = data.reset_index()

# Print column names for debugging
# st.write("Columns in data:", data.columns)

# Handle multi-index columns if necessary
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.map(' '.join).str.strip()

# Locate the Close column dynamically
close_column = [col for col in data.columns if 'Close' in col][0]
st.write(f"Using column '{close_column}' for Close prices")

# Keep only the Close column and rename it
data = data[['Date', close_column]].rename(columns={close_column: 'Close'})

# Display the cleaned data
st.write(data)

# Bitcoin Line Chart
st.subheader('Bitcoin Line Chart')
st.line_chart(data.set_index('Date')['Close'])

# Prepare data for training/testing
train_data = data['Close'][:-100].values.reshape(-1, 1)
test_data = data['Close'][-200:].values.reshape(-1, 1)

# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
train_data_scale = scaler.fit_transform(train_data)
test_data_scale = scaler.transform(test_data)

# Prepare input sequences for the model
base_days = 100
x = []
y = []
for i in range(base_days, len(test_data_scale)):
    x.append(test_data_scale[i - base_days:i])
    y.append(test_data_scale[i, 0])

x, y = np.array(x), np.array(y)
x = np.reshape(x, (x.shape[0], x.shape[1], 1))

# Predicted vs Original Prices
st.subheader('Predicted vs Original Prices')
pred = model.predict(x)
pred = scaler.inverse_transform(pred)
preds = pred.reshape(-1, 1)
ys = scaler.inverse_transform(y.reshape(-1, 1))
preds = pd.DataFrame(preds, columns=['Predicted Price'])
ys = pd.DataFrame(ys, columns=['Original Price'])
chart_data = pd.concat((preds, ys), axis=1)

# Display predictions
st.write(chart_data)
st.subheader('Predicted vs Original Prices Chart')
st.line_chart(chart_data)

# Predict future prices
m = test_data_scale[-base_days:]
z = []
future_days = 5
for _ in range(future_days):
    inter = m[-base_days:].reshape(1, base_days, 1)
    pred = model.predict(inter)
    z.append(pred[0, 0])
    m = np.append(m, pred)

st.subheader('Predicted Future Days Bitcoin Price')
z = np.array(z).reshape(-1, 1)
z = scaler.inverse_transform(z)
st.line_chart(z)
