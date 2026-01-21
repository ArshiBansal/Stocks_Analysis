# ==================== IMPORTS ====================
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor

from statsmodels.tsa.arima.model import ARIMA

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# ==================== OPTIONAL PROPHET ====================
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except:
    PROPHET_AVAILABLE = False

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="Magnificent 7+ AI Forecaster Pro",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== STYLING ====================
st.markdown("""
<style>
.main-header {
    font-size: 3.2rem;
    font-weight: 800;
    text-align: center;
    background: linear-gradient(90deg, #00ff88, #00d4ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.sub-header {
    font-size: 1.6rem;
    color: #00ff88;
    font-weight: 600;
    margin: 1.2rem 0;
}
</style>
""", unsafe_allow_html=True)

# ==================== SIDEBAR ====================
stocks = {
    "Magnificent 7": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"],
    "Top Tech Giants": ["AMD", "NFLX", "ADBE", "ORCL", "CRM", "INTC"]
}

st.sidebar.markdown("## 🤖 AI Forecaster Pro")
category = st.sidebar.selectbox("Stock Group", list(stocks.keys()))
ticker = st.sidebar.selectbox("Select Stock", stocks[category])
forecast_days = st.sidebar.slider("Forecast Days", 10, 180, 60, 10)
lookback_days = st.sidebar.slider("Lookback Window", 30, 180, 90, 10)

# ==================== CACHE RESET ====================
if st.sidebar.button("Clear Cache & Full Refresh"):
    st.cache_data.clear()
    st.session_state.clear()
    st.rerun()

# ==================== DATA LOADER (CLOUD SAFE) ====================
@st.cache_data(ttl=3600)
def load_data(ticker):
    df = yf.download(
        ticker,
        period="5y",
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=False
    )
    if not df.empty:
        df.reset_index(inplace=True)
    return df

df = load_data(ticker)

if df.empty:
    st.error("❌ Yahoo Finance blocked this request. Try later or another ticker.")
    st.stop()

# ==================== FEATURE ENGINEERING ====================
df["SMA_50"] = df["Close"].rolling(50).mean()
df["SMA_200"] = df["Close"].rolling(200).mean()
df["Return"] = df["Close"].pct_change()
df["Volatility_30d"] = df["Return"].rolling(30).std() * np.sqrt(252)
df["Cumulative_Return"] = (1 + df["Return"]).cumprod() - 1

# ==================== HEADER ====================
latest = df.iloc[-1]
change = (latest.Close - df.Close.iloc[-2]) / df.Close.iloc[-2] * 100

st.markdown(f"<h1 class='main-header'>{ticker} Stock Analysis</h1>", unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)
c1.metric("Price", f"${latest.Close:.2f}", f"{change:+.2f}%")
c2.metric("Volume", f"{latest.Volume/1e6:.1f}M")
c3.metric("52W High", f"${df.High.max():.2f}")

st.markdown("---")

# ==================== TABS ====================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Overview", "Raw Data", "Time Series + ML",
    "Neural Networks", "Performance", "AI Forecast"
])

# ==================== TAB 1: OVERVIEW ====================
with tab1:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=df.Date, y=df.Close, name="Close", line=dict(color="#00ff88", width=3)))
    fig.add_trace(go.Scatter(x=df.Date, y=df.SMA_50, name="SMA 50", line=dict(dash="dash")))
    fig.add_trace(go.Scatter(x=df.Date, y=df.SMA_200, name="SMA 200", line=dict(dash="dot")))
    fig.add_trace(go.Bar(x=df.Date, y=df.Volume, name="Volume", opacity=0.25), secondary_y=True)
    fig.update_layout(template="plotly_dark", height=500)
    st.plotly_chart(fig, use_container_width=True)

# ==================== TAB 2: RAW DATA ====================
with tab2:
    st.dataframe(df.tail(200), use_container_width=True)
    st.download_button("Download CSV", df.to_csv(index=False).encode(), f"{ticker}.csv")

# ==================== TAB 3: TS + ML ====================
with tab3:
    if st.button("Train Models"):
        models, backtests = {}, {}

        series = df["Close"]
        arima = ARIMA(series, order=(5,1,0)).fit()
        steps = len(series)//5
        pred = arima.forecast(steps)
        backtests["ARIMA"] = (df.Date.iloc[-steps:], series.iloc[-steps:], pred)
        models["ARIMA"] = arima

        if PROPHET_AVAILABLE:
            p_df = df[["Date","Close"]].rename(columns={"Date":"ds","Close":"y"})
            m = Prophet()
            m.fit(p_df)
            fc = m.predict(m.make_future_dataframe(0))
            backtests["Prophet"] = (df.Date.iloc[-steps:], series.iloc[-steps:], fc.yhat.values[-steps:])
            models["Prophet"] = m

        lagged = pd.DataFrame({f"lag_{i}": series.shift(i) for i in range(1,21)})
        lagged["y"] = series
        lagged.dropna(inplace=True)

        X = lagged.drop("y", axis=1)
        y = lagged["y"]
        split = int(0.8*len(X))

        rf = RandomForestRegressor(n_estimators=200, random_state=42)
        rf.fit(X[:split], y[:split])
        pred = rf.predict(X[split:])
        backtests["Random Forest"] = (df.Date.iloc[-len(pred):], y[split:], pred)
        models["Random Forest"] = rf

        st.session_state.models_ts_ml = models
        st.session_state.backtest_ts_ml = backtests
        st.success("Models trained")

# ==================== TAB 4: LSTM + GRU ====================
with tab4:
    if st.button("Train LSTM + GRU"):
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(df.Close.values.reshape(-1,1))

        def make_seq(data, n):
            X, y = [], []
            for i in range(n, len(data)):
                X.append(data[i-n:i])
                y.append(data[i])
            return np.array(X), np.array(y)

        X, y = make_seq(scaled, lookback_days)
        split = int(0.8*len(X))

        model = Sequential([
            LSTM(80, return_sequences=True, input_shape=(lookback_days,1)),
            Dropout(0.3),
            LSTM(80),
            Dense(1)
        ])
        model.compile("adam","mse")
        model.fit(X[:split], y[:split], epochs=30, batch_size=32, verbose=0)

        st.session_state.lstm_model = (model, scaler)
        st.success("LSTM trained")

# ==================== TAB 5: METRICS ====================
with tab5:
    rows = []
    if "backtest_ts_ml" in st.session_state:
        for name,(d,a,p) in st.session_state.backtest_ts_ml.items():
            rows.append({
                "Model":name,
                "MAE":mean_absolute_error(a,p),
                "RMSE":np.sqrt(mean_squared_error(a,p)),
                "R2":r2_score(a,p)
            })
    if rows:
        st.dataframe(pd.DataFrame(rows).round(3), use_container_width=True)

# ==================== TAB 6: FORECAST ====================
with tab6:
    if "lstm_model" not in st.session_state:
        st.warning("Train models first")
    else:
        model, scaler = st.session_state.lstm_model
        seq = scaler.transform(df.Close.values[-lookback_days:].reshape(-1,1))
        preds = []
        cur = seq.copy()
        for _ in range(forecast_days):
            p = model.predict(cur.reshape(1,lookback_days,1), verbose=0)[0,0]
            preds.append(p)
            cur = np.append(cur[1:], [[p]], axis=0)

        forecast = scaler.inverse_transform(np.array(preds).reshape(-1,1)).flatten()
        future_dates = pd.bdate_range(df.Date.iloc[-1]+timedelta(days=1), periods=forecast_days)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.Date, y=df.Close, name="Historical"))
        fig.add_trace(go.Scatter(x=future_dates, y=forecast, name="LSTM Forecast"))
        fig.update_layout(template="plotly_dark", height=500)
        st.plotly_chart(fig, use_container_width=True)

        out = pd.DataFrame({"Date":future_dates,"Forecast":forecast})
        st.dataframe(out, use_container_width=True)
