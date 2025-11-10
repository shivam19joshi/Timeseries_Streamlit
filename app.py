import streamlit as st
import yfinance as yf
import datetime as d
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
import plotly.graph_objects as go

st.set_page_config(page_title="Stock Forecast Lab", layout="wide")

# ---------------------------
# DOWNLOAD DATA
# ---------------------------
s = d.datetime(2024,1,1)
e = d.datetime(2025,11,11)

stocks = {
    "TCS" : "TCS.NS",
    "WIPRO" : "WIPRO.NS",
    "HCLTECH" : "HCLTECH.NS"
}

# sidebar config
st.sidebar.title("Stock Forecast Lab")

stock_select = st.sidebar.selectbox("Choose Stock", list(stocks.keys()))
p = st.sidebar.slider("AR (p)", 0,10,5)
d1 = st.sidebar.slider("I (d)", 0,2,1)
q = st.sidebar.slider("MA (q)", 0,10,0)

steps = st.sidebar.slider("Forecast Horizon (days)",5,60,10)

data = yf.download(stocks[stock_select], start=s, end=e)
data = data.reset_index()
data = data.set_index("Date")
close = data["Close"]

# tabs
tab1, tab2, tab3 = st.tabs(["📈 Chart","📉 Stationarity Test","📄 Raw Data"])

with tab2:
    st.subheader("ADF Test (Stationarity)")
    result = adfuller(close.dropna())
    pvalue = result[1]
    st.write("ADF p-value:", round(pvalue,4))
    if pvalue <= 0.05:
        st.success("Series Stationary ✅")
    else:
        st.error("Series NOT Stationary ❌")

# model
model = ARIMA(close, order=(p,d1,q))
model_fit = model.fit()
forecast = model_fit.forecast(steps=steps)
future_dates = pd.date_range(start=close.index[-1], periods=steps+1, freq='B')[1:]

# chart
with tab1:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=close.index, y=close, name="Actual", mode='lines'))
    fig.add_trace(go.Scatter(x=future_dates, y=forecast, name="Forecast", mode='lines', line=dict(dash='dash')))
    fig.update_layout(title=f"{stock_select} Forecast", xaxis_title="Date", yaxis_title="Price")
    st.plotly_chart(fig, use_container_width=True)

# raw data
with tab3:
    st.dataframe(data)
