import streamlit as st
import yfinance as yf
import datetime as d
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import plotly.graph_objects as go

st.set_page_config(page_title="Stock Forecast", layout="wide")

#---------------------------------------
# DATA DOWNLOAD
#---------------------------------------
s = d.datetime(2024,1,1)
e = d.datetime(2025,11,11)

tcs = yf.download('TCS.NS',start=s,end=e)
tcs.columns = tcs.columns.get_level_values(0)
tcs = tcs.reset_index()
tcs['Stock'] = 'TCS'

wipro = yf.download('WIPRO.NS',start=s,end=e)
wipro.columns = wipro.columns.get_level_values(0)
wipro = wipro.reset_index()
wipro['Stock'] = 'WIPRO'

hcl = yf.download('HCLTECH.NS',start=s,end=e)
hcl.columns = hcl.columns.get_level_values(0)
hcl = hcl.reset_index()
hcl['Stock'] = 'HCLTECH'

df = pd.concat([tcs,wipro,hcl],axis=0)
df = df.set_index('Date')

st.sidebar.title("Stock Forecast App")

stock_list = df.Stock.unique().tolist()
stock_select = st.sidebar.selectbox("Select Stock", stock_list)

st_df = df[df.Stock == stock_select][['Close']]
st_df['Returns'] = st_df['Close'].pct_change()
st_df.dropna(inplace=True)

def check(timeseries):
    result = adfuller(timeseries.dropna())
    return result[1]

pValue = check(st_df['Close'])
st.sidebar.write("ADF p-value:", round(pValue,4))
if pValue <= 0.05:
    st.sidebar.success("Series is Stationary ✅")
else:
    st.sidebar.error("Series is NOT Stationary ❌")

# DIFF
st_df['Close_Diff'] = st_df['Close'].diff().dropna()
p2 = check(st_df['Close_Diff'])
st.sidebar.write("ADF After Diff:", round(p2,4))


#---------------------------------------
# MODEL
#---------------------------------------
model = ARIMA(st_df['Close'], order=(5,1,0))
model_fit = model.fit()
forecast = model_fit.forecast(steps=10)
future_dates = pd.date_range(start=st_df.index[-1], periods=11, freq='B')[1:]

#---------------------------------------
# PLOTLY GRAPH
#---------------------------------------
fig = go.Figure()
fig.add_trace(go.Scatter(x=st_df.index, y=st_df['Close'], mode='lines', name='Actual Price'))
fig.add_trace(go.Scatter(x=future_dates, y=forecast, mode='lines', name='Predicted Price', line=dict(dash='dash')))

fig.update_layout(
    title=f"{stock_select} Stock Prediction",
    xaxis_title="Date",
    yaxis_title="Price"
)

st.plotly_chart(fig, use_container_width=True)
