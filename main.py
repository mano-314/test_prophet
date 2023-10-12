import streamlit as st 
import plotly.express as px
import pandas as pd
import yfinance as yf
import pandas as pd
import plotly.express as px
from prophet import Prophet
from yahooquery import Ticker

a = st.text_input("input myport")
st.write(a)
# st.plotly_chart(px)

myport = [a]  # list of tickers you want to analyze
start_date = "2022-01-01"
end_date = "2023-10-04"
df = yf.download(myport, start=start_date, end=end_date)
df = df.reset_index()
columns = ['Date','Close']
ndf = pd.DataFrame(df, columns = columns)
prophet_df = ndf.rename(columns={'Date':'ds','Close':'y'})

m = Prophet()
m.fit(prophet_df) #train data
future = m.make_future_dataframe(periods=30)

forecast = m.predict(future)
forecast
px.line(forecast, x='ds', y='yhat')
figure = m.plot(forecast,xlabel="ds",ylabel="y") #black dot = price of stock
figure2 = m.plot_components(forecast)

st.plotly_chart(figure2)
