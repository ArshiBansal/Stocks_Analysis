# --------------------------------------------------------------
#  Zephyr – AI-Powered Stock Intelligence + NEWSLETTER
#  Subscribe in Tab 7 → Get Daily News Email
#  NO RERUN ON TRAIN | ALL ERRORS FIXED
# --------------------------------------------------------------

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
from prophet import Prophet
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, GRU, Input
from tensorflow.keras.callbacks import EarlyStopping
from lightgbm import LGBMRegressor
import requests
from bs4 import BeautifulSoup
import time
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# === VADER SENTIMENT (Auto-install) ===
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    analyzer = SentimentIntensityAnalyzer()
except Exception:
    st.warning("Installing vaderSentiment...")
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "vaderSentiment"])
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    analyzer = SentimentIntensityAnalyzer()

# ==================== PAGE CONFIG & THEME ====================
st.set_page_config(page_title="Zephyr", page_icon="Wind", layout="wide", initial_sidebar_state="expanded")

# ==================== LIVE PRICE TICKER ====================
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = 0

current_time = time.time()
if current_time - st.session_state.last_refresh > 30:
    st.session_state.last_refresh = current_time

@st.cache_data(ttl=25)
def get_live_price(ticker):
    try:
        data = yf.Ticker(ticker).history(period="1d", interval="1m")
        if not data.empty:
            return data['Close'].iloc[-1]
    except:
        pass
    return None

# ==================== CSS STYLING ====================
st.markdown("""
<style>
    .main-header {font-size: 3.5rem; font-weight: 800; text-align: center;
        background: linear-gradient(90deg, #00ff88, #00d4ff, #7c3aed);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;}
    .sub-header {font-size: 1.8rem; color: #00ff88; font-weight: 600; margin: 1.5rem 0;}
    .stTabs [data-baseweb="tab"] {background-color: #16213e; border-radius: 10px 10px 0 0; padding: 14px 28px; color: #e0e0e0;}
    .stTabs [aria-selected="true"] {background-color: #00ff881a !important; border-bottom: 4px solid #00ff88; color: #00ff88;}
    .css-1d391kg {background-color: #0f0f1e;}
    .news-card {background: #1a1a2e; padding: 16px; border-radius: 12px; margin: 10px 0; border-left: 4px solid #00ff88;}
    .sentiment-pos {color: #00ff88; font-weight: bold;}
    .sentiment-neg {color: #ff00aa; font-weight: bold;}
    .sentiment-neu {color: #888;}
    .section-header {font-size: 1.4rem; color: #00ff88; font-weight: 700; margin: 1.5rem 0 0.5rem; padding-bottom: 0.3rem; border-bottom: 1px solid #00ff8830;}
    .subscribe-box {background: #1a1a2e; padding: 20px; border-radius: 12px; border: 1px solid #00ff88; text-align: center; margin: 20px 0;}
</style>
""", unsafe_allow_html=True)

# ==================== STOCKS & SIDEBAR ====================
stocks = {
    "Magnificent 7": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"],
    "Top Tech Giants": ["AMD", "NFLX", "ADBE", "ORCL", "CRM", "INTC", "PYPL", "AVGO", "QCOM", "TXN"],
    "AI Chip & Cloud Leaders": ["TSM", "ASML", "SNOW", "CRWD", "PLTR", "ARM"]
}

st.sidebar.markdown("<h2 style='color:#00ff88;'>Zephyr</h2><p style='color:#888;'>8 Models • Ensemble • Live News</p>", unsafe_allow_html=True)
category = st.sidebar.selectbox("Stock Group", list(stocks.keys()))
ticker = st.sidebar.selectbox("Select Stock", stocks[category])
forecast_days = st.sidebar.slider("Forecast Days", 10, 180, 60, 10)
lookback_days = st.sidebar.slider("Lookback Window", 30, 180, 90, 10)

TRAIN_LOOKBACK = 60
BACKTEST_DAYS = 252

# ==================== CLEAR CACHE ====================
if st.sidebar.button("Clear **ALL** Cache & Refresh", use_container_width=True):
    st.cache_data.clear()
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

# ==================== LOAD DATA ====================
@st.cache_data(ttl=300)
def load_data(ticker):
    with st.spinner(f"Fetching {ticker} data..."):
        data = yf.Ticker(ticker).history(period="5y", auto_adjust=True)
        if not data.empty and data.index.tz is not None:
            data.index = data.index.tz_localize(None)
        data.reset_index(inplace=True)
        return data

df = load_data(ticker)
if df.empty:
    st.error("No data found!")
    st.stop()

# ==================== LIVE PRICE IN SIDEBAR ====================
st.sidebar.markdown("---")
st.sidebar.markdown("### Live Price")
live_price = get_live_price(ticker)
if live_price is None:
    live_price = st.session_state.get("last_close", df['Close'].iloc[-1])
else:
    st.session_state.last_close = live_price

price_change = ((live_price - df['Close'].iloc[-2]) / df['Close'].iloc[-2] * 100) \
    if len(df) > 1 else 0
delta_color = "normal" if price_change >= 0 else "inverse"
st.sidebar.metric("Current Price", f"${live_price:.2f}", f"{price_change:+.2f}%", delta_color=delta_color)
st.sidebar.caption(f"Updated now • Next in ~{int(30 - (current_time - st.session_state.last_refresh))}s")

# ==================== DATA PREP & HEADER ====================
if 'last_data_fetch' not in st.session_state:
    st.session_state.last_data_fetch = datetime.now()

latest_trading_day = df['Date'].iloc[-1].strftime("%B %d, %Y")
fetch_time = st.session_state.last_data_fetch.strftime("%I:%M %p")

st.markdown(f"""
<div style='text-align: center; background: rgba(0,255,136,0.1); padding: 18px; border-radius: 12px; margin: 20px 0; border: 1px solid #00ff88;'>
    <p style='margin:0; color:#00ff88; font-size:1.35rem; font-weight:600;'>
        Latest Trading Day: {latest_trading_day}
    </p>
    <p style='margin:8px 0 0; color:#bbbbbb; font-size:1rem;'>
        Data refreshed today at {fetch_time}
    </p>
</div>
""", unsafe_allow_html=True)

info = yf.Ticker(ticker).info
df['SMA_50'] = df['Close'].rolling(50).mean()
df['SMA_200'] = df['Close'].rolling(200).mean()
df['Return'] = df['Close'].pct_change()
df['Volatility_30d'] = df['Return'].rolling(30).std() * np.sqrt(252)
df['Cumulative_Return'] = (1 + df['Return']).cumprod() - 1

st.markdown(f'<h1 class="main-header">{ticker} — {info.get("longName","")}</h1>', unsafe_allow_html=True)
col1, col2, col3, col4, col5, col6 = st.columns(6)
latest = df.iloc[-1]
change = (latest.Close - df['Close'].iloc[-2]) / df['Close'].iloc[-2] * 100 if len(df) > 1 else 0

with col1: st.metric("Price", f"${latest.Close:.2f}", f"{change:+.2f}%")
with col2: st.metric("Volume", f"{latest.Volume/1e6:.1f}M")
with col3: st.metric("Market Cap", f"${info.get('marketCap',0)/1e12:.2f}T")
with col4: st.metric("52W High", f"${df['High'].max():.2f}")
with col5: st.metric("52W Low", f"${df['Low'].min():.2f}")
with col6: st.metric("P/E Ratio", f"{info.get('trailingPE','N/A'):.2f}" if info.get('trailingPE') else "N/A")

st.markdown("---")

# ==================== TABS ====================
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Overview", "Raw Data", "Time Series + ML", "Neural Networks", "Performance Metrics", "AI Forecast", "News & Sentiment"
])

# ==================== TAB 1: OVERVIEW ====================
with tab1:
    st.markdown('<div class="sub-header">Comprehensive Stock Overview</div>', unsafe_allow_html=True)

    fig1 = make_subplots(specs=[[{"secondary_y": True}]])
    fig1.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name="Close Price", line=dict(color="#00ff88", width=3)))
    fig1.add_trace(go.Scatter(x=df['Date'], y=df['SMA_50'], name="SMA 50", line=dict(dash="dash", color="#ffa500")))
    fig1.add_trace(go.Scatter(x=df['Date'], y=df['SMA_200'], name="SMA 200", line=dict(dash="dot", color="#7c3aed")))
    fig1.add_trace(go.Bar(x=df['Date'], y=df['Volume'], name="Volume", opacity=0.25, marker_color="#333333"), secondary_y=True)
    fig1.update_layout(title="Price Trend with Moving Averages & Volume", template="plotly_dark", height=500)
    fig1.update_yaxes(title_text="Price ($)", secondary_y=False)
    fig1.update_yaxes(title_text="Volume", secondary_y=True)
    st.plotly_chart(fig1, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        fig2 = go.Figure()
        fig2.add_trace(go.Histogram(x=df['Return'].dropna() * 100, nbinsx=70, name="Daily Returns", marker_color="#7c3aed", opacity=0.8))
        fig2.update_layout(title="Daily Returns Distribution (%)", template="plotly_dark", height=400, xaxis_title="Return (%)", yaxis_title="Frequency")
        st.plotly_chart(fig2, use_container_width=True)
    with col_b:
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=df['Date'], y=df['Volatility_30d'], name="30D Volatility", line=dict(color="#ff00aa", width=2)))
        fig3.update_layout(title="30-Day Rolling Volatility (Annualized)", template="plotly_dark", height=400)
        st.plotly_chart(fig3, use_container_width=True)

    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(x=df['Date'], y=df['Cumulative_Return'] * 100, name="Cumulative Return %", line=dict(color="rgba(0, 255, 136, 0.9)", width=3)))
    fig4.add_hline(y=0, line_dash="dash", line_color="gray")
    fig4.update_layout(title="Cumulative Returns (Buy & Hold)", template="plotly_dark", height=450, yaxis_title="Total Return (%)")
    st.plotly_chart(fig4, use_container_width=True)

# ==================== TAB 2: RAW DATA ====================
with tab2:
    st.markdown('<div class="sub-header">Historical Data</div>', unsafe_allow_html=True)
    st.dataframe(df.tail(200), use_container_width=True)
    st.download_button("Download Historical Data", df.to_csv(index=False).encode(), f"{ticker}_zephyr_history.csv", "text/csv")

# ==================== TAB 3: TIME SERIES + ML (NO RERUN) ====================
with tab3:
    st.markdown('<div class="sub-header">Train 6 Classic ML Models + Backtest</div>', unsafe_allow_html=True)

    if st.button("Clear ML Models Only", use_container_width=True):
        for k in ['ml_models', 'ml_backtest', 'ml_training_done']:
            st.session_state.pop(k, None)
        st.success("ML models cleared!")

    if 'ml_training_done' not in st.session_state:
        st.session_state.ml_training_done = False

    if st.button("Train All 6 Models", type="primary", use_container_width=True, key="train_ml_btn"):
        st.session_state.ml_training_done = True
        with st.spinner("Training 6 ML models..."):
            models = {}
            backtest_data = {}
            test_size = min(BACKTEST_DAYS, len(df) // 4)

            lagged = df[['Close']].copy()
            for i in range(1, 21):
                lagged[f'lag_{i}'] = lagged['Close'].shift(i)
            lagged['target'] = lagged['Close']
            lagged.dropna(inplace=True)

            split = len(lagged) - test_size
            X_train, X_test = lagged.drop('target', axis=1).iloc[:split], lagged.drop('target', axis=1).iloc[split:]
            y_train, y_test = lagged['target'].iloc[:split], lagged['target'].iloc[split:]
            test_dates = df['Date'].iloc[-test_size:].values

            with st.expander("1. LightGBM"):
                model = LGBMRegressor(n_estimators=500, learning_rate=0.05, random_state=42, n_jobs=-1)
                model.fit(X_train, y_train)
                pred = model.predict(X_test)
                backtest_data['LightGBM'] = (test_dates, y_test.values, pred)
                models['LightGBM'] = model
                st.success("LightGBM trained")

            with st.expander("2. Prophet"):
                p_df = df[['Date', 'Close']].copy()
                p_df.columns = ['ds', 'y']
                p_df = p_df.iloc[:-test_size]
                p_df['ds'] = pd.to_datetime(p_df['ds']).dt.tz_localize(None)
                m = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
                m.add_seasonality(name='monthly', period=30.5, fourier_order=5)
                m.fit(p_df)
                future = m.make_future_dataframe(periods=test_size, freq='B')
                fc = m.predict(future)
                pred = fc['yhat'].values[-test_size:]
                backtest_data['Prophet'] = (df['Date'].iloc[-test_size:].values, df['Close'].iloc[-test_size:].values, pred)
                models['Prophet'] = m
                st.success("Prophet trained")

            with st.expander("3. Random Forest"):
                rf = RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=42)
                rf.fit(X_train, y_train)
                pred = rf.predict(X_test)
                backtest_data['Random Forest'] = (test_dates, y_test.values, pred)
                models['Random Forest'] = rf
                st.success("RF trained")

            with st.expander("4. XGBoost"):
                xgb = XGBRegressor(n_estimators=500, learning_rate=0.05, n_jobs=-1, random_state=42)
                xgb.fit(X_train, y_train)
                pred = xgb.predict(X_test)
                backtest_data['XGBoost'] = (test_dates, y_test.values, pred)
                models['XGBoost'] = xgb
                st.success("XGBoost trained")

            with st.expander("5. K-Means"):
                features = df[['Close', 'Volume', 'Return', 'Volatility_30d']].fillna(0).iloc[-test_size:]
                kmeans = KMeans(n_clusters=4, random_state=42, n_init=10).fit(features)
                pred = np.array([kmeans.cluster_centers_[l, 0] for l in kmeans.labels_])
                backtest_data['K-Means'] = (test_dates, features['Close'].values, pred)
                models['K-Means'] = kmeans
                st.success("K-Means trained")

            with st.expander("6. Autoencoder"):
                input_dim = X_train.shape[1]
                input_layer = Input(shape=(input_dim,))
                encoded = Dense(32, activation='relu')(input_layer)
                encoded = Dense(16, activation='relu')(encoded)
                decoded = Dense(32, activation='relu')(encoded)
                decoded = Dense(input_dim, activation='linear')(decoded)
                autoencoder = Model(input_layer, decoded)
                autoencoder.compile(optimizer='adam', loss='mse')
                autoencoder.fit(X_train, X_train, epochs=50, batch_size=32, verbose=0, callbacks=[EarlyStopping(patience=5)])
                recon = autoencoder.predict(X_test)
                mse = np.mean(np.power(X_test - recon, 2), axis=1)
                threshold = np.percentile(mse, 98)
                pred = np.where(mse > threshold, y_test * 0.9, y_test)
                backtest_data['Autoencoder'] = (test_dates, y_test.values, pred)
                models['Autoencoder'] = autoencoder
                st.success("Autoencoder trained")

            st.session_state.ml_models = models
            st.session_state.ml_backtest = backtest_data
            st.success("All 6 ML models trained!")

    if st.session_state.get("ml_backtest"):
        model_name = st.selectbox("Select ML Model", [""] + list(st.session_state.ml_backtest.keys()), key="ml_select")
        if model_name:
            dates_raw, actual, pred = st.session_state.ml_backtest[model_name]
            dates = pd.to_datetime(dates_raw).strftime("%Y-%m-%d")
            results = pd.DataFrame({"Date": dates, "Actual": np.round(actual, 2), "Predicted": np.round(pred, 2)})
            st.markdown(f"#### {model_name} — Backtest")
            st.dataframe(results.style.format({"Actual": "${:.2f}", "Predicted": "${:.2f}"}), use_container_width=True)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=dates, y=actual, name="Actual", line=dict(color="white")))
            fig.add_trace(go.Scatter(x=dates, y=pred, name="Predicted", line=dict(color="#00ff88", dash="dot")))
            fig.update_layout(template="plotly_dark", height=450, title=model_name)
            st.plotly_chart(fig, use_container_width=True)

# ==================== TAB 4: NEURAL NETWORKS (NO RERUN) ====================
with tab4:
    st.markdown('<div class="sub-header">Train LSTM & GRU + Backtest</div>', unsafe_allow_html=True)

    if st.button("Clear DL Models Only", use_container_width=True):
        for k in ['dl_backtest', 'dl_lstm_model', 'dl_gru_model', 'dl_training_done']:
            st.session_state.pop(k, None)
        st.success("DL models cleared!")

    if 'dl_training_done' not in st.session_state:
        st.session_state.dl_training_done = False

    if st.button("Train LSTM + GRU", type="primary", use_container_width=True, key="train_dl_btn"):
        st.session_state.dl_training_done = True
        with st.spinner("Training LSTM & GRU..."):
            scaler = MinMaxScaler()
            scaled = scaler.fit_transform(df['Close'].values.reshape(-1, 1))

            def create_dataset(data, steps):
                X, y = [], []
                for i in range(steps, len(data)):
                    X.append(data[i-steps:i])
                    y.append(data[i])
                return np.array(X), np.array(y)

            X, y = create_dataset(scaled, TRAIN_LOOKBACK)
            X = X.reshape((X.shape[0], X.shape[1], 1))
            split = len(X) - BACKTEST_DAYS
            test_dates = df['Date'].iloc[split + TRAIN_LOOKBACK:].values

            backtest_dl = {}

            with st.expander("LSTM"):
                model = Sequential([
                    LSTM(100, return_sequences=True, input_shape=(TRAIN_LOOKBACK, 1)), Dropout(0.3),
                    LSTM(100), Dropout(0.3), Dense(1)
                ])
                model.compile(optimizer='adam', loss='mse')
                model.fit(X[:split], y[:split], epochs=50, batch_size=32, verbose=0, callbacks=[EarlyStopping(patience=10)])
                pred_scaled = model.predict(X[split:], verbose=0)
                pred = scaler.inverse_transform(pred_scaled).flatten()
                actual = scaler.inverse_transform(y[split:]).flatten()
                backtest_dl['LSTM'] = (test_dates, actual, pred)
                st.session_state.dl_lstm_model = (model, scaler)
                st.success("LSTM trained")

            with st.expander("GRU"):
                model = Sequential([
                    GRU(100, return_sequences=True, input_shape=(TRAIN_LOOKBACK, 1)), Dropout(0.3),
                    GRU(100), Dropout(0.3), Dense(1)
                ])
                model.compile(optimizer='adam', loss='mse')
                model.fit(X[:split], y[:split], epochs=50, batch_size=32, verbose=0, callbacks=[EarlyStopping(patience=10)])
                pred_scaled = model.predict(X[split:], verbose=0)
                pred = scaler.inverse_transform(pred_scaled).flatten()
                actual = scaler.inverse_transform(y[split:]).flatten()
                backtest_dl['GRU'] = (test_dates, actual, pred)
                st.session_state.dl_gru_model = (model, scaler)
                st.success("GRU trained")

            st.session_state.dl_backtest = backtest_dl
            st.success("Deep learning models trained!")

    if st.session_state.get("dl_backtest"):
        dl_model = st.selectbox("Select DL Model", [""] + list(st.session_state.dl_backtest.keys()), key="dl_select")
        if dl_model:
            dates_raw, actual, pred = st.session_state.dl_backtest[dl_model]
            dates = pd.to_datetime(dates_raw).strftime("%Y-%m-%d")
            results = pd.DataFrame({"Date": dates, "Actual": np.round(actual, 2), "Predicted": np.round(pred, 2)})
            st.markdown(f"#### {dl_model} — Backtest")
            st.dataframe(results.style.format({"Actual": "${:.2f}", "Predicted": "${:.2f}"}), use_container_width=True)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=dates, y=actual, name="Actual", line=dict(color="white")))
            fig.add_trace(go.Scatter(x=dates, y=pred, name="Predicted", line=dict(color="#7c3aed", dash="dot")))
            fig.update_layout(template="plotly_dark", height=450)
            st.plotly_chart(fig, use_container_width=True)

# ==================== TAB 5: PERFORMANCE METRICS ====================
with tab5:
    st.markdown('<div class="sub-header">Model Performance — 5 Key Metrics</div>', unsafe_allow_html=True)
    metrics_list = []

    def add_metrics(name, actual, pred):
        mae = mean_absolute_error(actual, pred)
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mape = np.mean(np.abs((actual - pred) / (actual + 1e-8))) * 100
        r2 = r2_score(actual, pred)
        directional = np.mean(np.sign(np.diff(actual)) == np.sign(np.diff(pred))) * 100 if len(actual) > 1 else 0
        metrics_list.append({"Model": name, "MAE": mae, "RMSE": rmse, "MAPE (%)": mape, "R² Score": r2, "Direction Accuracy (%)": directional})

    if st.session_state.get("ml_backtest"):
        for name, (_, actual, pred) in st.session_state.ml_backtest.items():
            add_metrics(name, actual, pred)
    if st.session_state.get("dl_backtest"):
        for name, (_, actual, pred) in st.session_state.dl_backtest.items():
            add_metrics(name, actual, pred)

    if metrics_list:
        metrics_df = pd.DataFrame(metrics_list).round({"MAE": 2, "RMSE": 2, "MAPE (%)": 2, "R² Score": 3, "Direction Accuracy (%)": 1})
        metrics_df = metrics_df.sort_values("MAE").reset_index(drop=True)
        metrics_df.index += 1
        st.session_state.metrics_df = metrics_df.copy()

        st.dataframe(metrics_df.style
                     .format({"MAE": "${:.2f}", "RMSE": "${:.2f}", "MAPE (%)": "{:.2f}%", "R² Score": "{:.3f}", "Direction Accuracy (%)": "{:.1f}%"})
                     .highlight_min(subset=["MAE", "RMSE", "MAPE (%)"], color='#00ff8820')
                     .highlight_max(subset=["R² Score", "Direction Accuracy (%)"], color='#00ff8820'),
                     use_container_width=True)

        best_model = metrics_df.sort_values("MAE").iloc[0]["Model"]
        st.success(f"Best Model: **{best_model}**")
    else:
        st.info("Train at least one model to see performance metrics")

# ==================== TAB 6: AI FORECAST WITH SIGNALS ====================
with tab6:
    st.markdown(f'<div class="sub-header">AI Forecast — Next {forecast_days} Days</div>', unsafe_allow_html=True)

    if not (st.session_state.get("ml_models") or st.session_state.get("dl_lstm_model")):
        st.warning("Train models first!")
    else:
        future_dates = pd.bdate_range(start=df['Date'].iloc[-1] + timedelta(days=1), periods=forecast_days)
        forecast_data = {"Date": future_dates.strftime("%Y-%m-%d")}
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name="Historical", line=dict(color="white", width=3)))

        if 'Prophet' in st.session_state.get("ml_models", {}):
            future = st.session_state.ml_models['Prophet'].make_future_dataframe(periods=forecast_days, freq='B')
            fc = st.session_state.ml_models['Prophet'].predict(future)
            pred = fc['yhat'][-forecast_days:].values
            forecast_data["Prophet"] = np.round(pred, 2)
            fig.add_trace(go.Scatter(x=future_dates, y=pred, name="Prophet", line=dict(color="#7c3aed", width=3)))

        if st.session_state.get("dl_lstm_model"):
            model, scaler = st.session_state.dl_lstm_model
            last_seq = scaler.transform(df['Close'].values[-lookback_days:].reshape(-1,1))
            preds = []
            cur = last_seq.copy()
            for _ in range(forecast_days):
                p = model.predict(cur.reshape(1, lookback_days, 1), verbose=0)[0,0]
                val = scaler.inverse_transform([[p]])[0,0]
                preds.append(val)
                cur = np.append(cur[1:], [[p]], axis=0)
            lstm_pred = np.array(preds)
            forecast_data["LSTM"] = np.round(lstm_pred, 2)
            fig.add_trace(go.Scatter(x=future_dates, y=lstm_pred, name="LSTM", line=dict(color="#00d4ff", dash="dot", width=3)))

        if len([k for k in forecast_data if k != "Date"]) > 1:
            pred_df = pd.DataFrame({k: v for k, v in forecast_data.items() if k != "Date"})
            if 'metrics_df' in st.session_state:
                weights = {}
                for col in pred_df.columns:
                    row = st.session_state.metrics_df[st.session_state.metrics_df['Model'] == col]
                    weights[col] = 1 / (row['MAE'].iloc[0] + 1e-8) if not row.empty else 1
                total = sum(weights.values())
                ensemble = sum(pred_df[col] * (weights[col] / total) for col in pred_df.columns)
            else:
                ensemble = pred_df.mean(axis=1)
            forecast_data['Ensemble'] = np.round(ensemble, 2)
            fig.add_trace(go.Scatter(x=future_dates, y=ensemble, name="Ensemble", line=dict(color="#FFD700", width=4)))

            ensemble_vals = pd.Series(forecast_data['Ensemble'])
            sma_5 = ensemble_vals.rolling(5).mean()
            sma_10 = ensemble_vals.rolling(10).mean()
            buy_signals = []
            sell_signals = []
            for i in range(1, len(ensemble_vals)):
                if sma_5.iloc[i] > sma_10.iloc[i] and sma_5.iloc[i-1] <= sma_10.iloc[i-1]:
                    buy_signals.append((future_dates[i], ensemble_vals[i]))
                elif sma_5.iloc[i] < sma_10.iloc[i] and sma_5.iloc[i-1] >= sma_10.iloc[i-1]:
                    sell_signals.append((future_dates[i], ensemble_vals[i]))

            if buy_signals:
                buy_x, buy_y = zip(*buy_signals)
                fig.add_trace(go.Scatter(x=buy_x, y=buy_y, mode='markers', name='BUY Signal',
                                         marker=dict(symbol='triangle-up', size=14, color='#00ff88', line=dict(width=2, color='white'))))
            if sell_signals:
                sell_x, sell_y = zip(*sell_signals)
                fig.add_trace(go.Scatter(x=sell_x, y=sell_y, mode='markers', name='SELL Signal',
                                         marker=dict(symbol='triangle-down', size=14, color='#ff00aa', line=dict(width=2, color='white'))))

            st.markdown("**Signal Logic**: Ensemble 5-day SMA crosses above 10-day SMA → **BUY** | Crosses below → **SELL**")

        fig.update_layout(template="plotly_dark", height=600, title=f"{ticker} — Multi-Model Forecast")
        st.plotly_chart(fig, use_container_width=True)

        forecast_df = pd.DataFrame(forecast_data)
        st.markdown("#### Forecast Values (Table)")
        st.dataframe(forecast_df.style.format({col: "${:.2f}" for col in forecast_df.columns if col != "Date"}), use_container_width=True)
        st.download_button("Download Forecast", forecast_df.to_csv(index=False).encode(),
                           f"{ticker}_zephyr_forecast_{forecast_days}d.csv", "text/csv", use_container_width=True)
        st.success("Forecast ready!")

# ==================== TAB 7: NEWS & SENTIMENT + SUBSCRIBE ====================
INVESTING_TICKER_MAP = {
    "AAPL": "apple", "MSFT": "microsoft", "GOOGL": "alphabet-a", "AMZN": "amazon-com-inc",
    "NVDA": "nvidia-corp", "META": "meta-platforms-inc", "TSLA": "tesla-motors",
    "AMD": "advanced-micro-devices", "NFLX": "netflix-inc", "ADBE": "adobe-inc",
    "ORCL": "oracle", "CRM": "salesforce-com", "INTC": "intel-corp", "PYPL": "paypal",
    "AVGO": "broadcom-inc", "QCOM": "qualcomm-inc", "TXN": "texas-instruments",
    "TSM": "taiwan-semiconductor", "ASML": "asml-holding", "SNOW": "snowflake-inc",
    "CRWD": "crowdstrike-holdings", "PLTR": "palantir-technologies", "ARM": "arm-holdings"
}

CATEGORIES = {
    "Market Movers": ["earnings", "revenue", "profit", "loss", "beat", "miss", "guidance", "upgrade", "downgrade", "price target", "analyst", "rating", "buy", "sell", "hold", "shares", "volume", "surge", "drop"],
    "Company Updates": ["launch", "product", "feature", "partnership", "acquisition", "merger", "CEO", "CFO", "executive", "hiring", "layoff", "expansion", "factory", "patent", "AI", "chip", "cloud", "software"],
    "Analyst & Macro": ["fed", "interest rate", "inflation", "recession", "GDP", "jobs", "unemployment", "market", "nasdaq", "dow", "S&P", "bull", "bear", "crash", "rally", "volatility", "VIX", "sector"]
}

@st.cache_data(ttl=3600)
def get_investing_news(ticker: str):
    articles = []
    search_term = INVESTING_TICKER_MAP.get(ticker, ticker.lower())
    url = f"https://www.investing.com/search/?q={search_term}"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

    try:
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, 'html.parser')
        items = soup.select(".articleItem")[:35]

        for it in items:
            a = it.find("a", class_="title")
            if not a: continue
            title = a.get("title") or a.get_text(strip=True)
            link = "https://www.investing.com" + a["href"]

            t_tag = it.find("time")
            pub_time = t_tag.get_text(strip=True) if t_tag else "Recent"
            try:
                pub_time = datetime.strptime(pub_time, "%b %d, %H:%M").strftime("%b %d, %I:%M %p")
            except: pass

            snip = it.find("p")
            snippet = (snip.get_text(strip=True)[:220] + "...") if snip else "Click to read..."

            vs = analyzer.polarity_scores(title + " " + snippet)
            score = vs['compound']
            if score >= 0.1:
                sentiment, cls = "Positive", "sentiment-pos"
            elif score <= -0.1:
                sentiment, cls = "Negative", "sentiment-neg"
            else:
                sentiment, cls = "Neutral", "sentiment-neu"

            text = (title + " " + snippet).lower()
            category = "Uncategorized"
            for cat, keywords in CATEGORIES.items():
                if any(k in text for k in keywords):
                    category = cat
                    break

            articles.append({
                "title": title, "link": link, "time": pub_time, "snippet": snippet,
                "sentiment": sentiment, "sent_class": cls, "score": score, "category": category
            })
    except Exception as e:
        st.error(f"News fetch failed: {e}")

    return articles

with tab7:
    st.markdown('<div class="sub-header">Latest News & Sentiment (Investing.com)</div>', unsafe_allow_html=True)

    with st.spinner("Fetching 30+ latest news articles..."):
        news_articles = get_investing_news(ticker)

    if not news_articles:
        st.info("No news available.")
    else:
        grouped = {"Market Movers": [], "Company Updates": [], "Analyst & Macro": [], "Uncategorized": []}
        for art in news_articles:
            grouped[art["category"]].append(art)

        scores = [a["score"] for a in news_articles if a["score"] != 0]
        avg = sum(scores) / len(scores) if scores else 0
        if avg > 0.1:
            overall = f"Overall Market Sentiment: <span style='color:#00ff88; font-weight:600;'>Bullish ({avg:+.2f})</span>"
        elif avg < -0.1:
            overall = f"Overall Market Sentiment: <span style='color:#ff00aa; font-weight:600;'>Bearish ({avg:+.2f})</span>"
        else:
            overall = f"Overall Market Sentiment: <span style='color:#888;'>Neutral ({avg:+.2f})</span>"
        st.markdown(f"<p style='text-align:center; font-size:1.3rem; padding:12px; background:#1a1a2e; border-radius:10px;'>{overall}</p>", unsafe_allow_html=True)

        for section in ["Market Movers", "Company Updates", "Analyst & Macro"]:
            arts = grouped[section]
            if not arts: continue
            with st.expander(f"{section} ({len(arts)} articles)", expanded=True):
                st.markdown(f"<div class='section-header'>{section}</div>", unsafe_allow_html=True)
                for art in arts:
                    st.markdown(f"""
                    <div class="news-card">
                        <p style='margin:0; font-weight:600; font-size:1.1rem;'>
                            <a href='{art["link"]}' target='_blank' style='color:#00d4ff; text-decoration:none;'>{art["title"]}</a>
                        </p>
                        <p style='margin:4px 0; color:#aaa; font-size:0.9rem;'>{art["time"]} • <span class='{art["sent_class"]}'>{art["sentiment"]}</span></p>
                        <p style='margin:4px 0 0; color:#ccc; font-size:0.95rem;'>{art["snippet"]}</p>
                    </div>
                    """, unsafe_allow_html=True)

        if grouped["Uncategorized"]:
            with st.expander(f"Other News ({len(grouped['Uncategorized'])})", expanded=False):
                for art in grouped["Uncategorized"]:
                    st.markdown(f"""
                    <div class="news-card">
                        <p style='margin:0; font-weight:600; font-size:1.1rem;'>
                            <a href='{art["link"]}' target='_blank' style='color:#00d4ff; text-decoration:none;'>{art["title"]}</a>
                        </p>
                        <p style='margin:4px 0; color:#aaa; font-size:0.9rem;'>{art["time"]} • <span class='{art["sent_class"]}'>{art["sentiment"]}</span></p>
                        <p style='margin:4px 0 0; color:#ccc; font-size:0.95rem;'>{art["snippet"]}</p>
                    </div>
                    """, unsafe_allow_html=True)

        # ==================== SUBSCRIBE BOX ====================
        st.markdown("---")
        st.markdown("### Get Daily Stock News in Your Inbox")
        st.markdown("""
        <div class="subscribe-box">
            <p style="color:#00ff88; font-weight:600; margin:0 0 10px;">Subscribe to Zephyr Daily Brief</p>
            <p style="color:#ccc; font-size:0.9rem; margin:5px 0 15px;">Top 5 news + sentiment + price alert</p>
        </div>
        """, unsafe_allow_html=True)

        email = st.text_input("Enter your email", placeholder="you@example.com", key="subscribe_email")
        col_sub, col_sp = st.columns([1, 3])
        with col_sub:
            subscribe = st.button("Subscribe Now", type="primary", use_container_width=True, key="subscribe_btn")

            if subscribe:
                if not email or "@" not in email or "." not in email:
                    st.error("Invalid email.")
            else:
                with st.spinner("Subscribing..."):
                    try:
                        scope = [
                            "https://spreadsheets.google.com/feeds",
                            "https://www.googleapis.com/auth/spreadsheets",
                            "https://www.googleapis.com/auth/drive"
                        ]
                        sa_dict = st.secrets["gcp_service_account"]
                        creds = ServiceAccountCredentials.from_json_keyfile_dict(sa_dict, scope)
                        client = gspread.authorize(creds)

                        # OPEN SHEET
                        sheet = client.open("Zephyr Subscribers").sheet1

                        # CHECK HEADER
                        header = sheet.row_values(1)
                        if header != ["email", "ticker", "timestamp"]:
                            st.error("Header must be: email, ticker, timestamp")
                            st.info(f"Found: {header}")
                            st.stop()

                        # CHECK DUPLICATE
                        emails = sheet.col_values(1)[1:]  # Skip header
                        if email in emails:
                            st.warning("Already subscribed!")
                        else:
                            sheet.append_row([email, ticker, datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
                            st.success(f"Subscribed {email}!")

                    except gspread.exceptions.SpreadsheetNotFound:
                        st.error("Sheet 'Zephyr Subscribers' not found!")
                        st.info("Create a sheet named exactly: **Zephyr Subscribers**")
                    except gspread.exceptions.APIError as e:
                        if "Resource not found" in str(e):
                            st.error("Sheet not shared with service account!")
                            st.code("Share with:\nzephyr-bot@stocks-480012.iam.gserviceaccount.com")
                        else:
                            st.error(f"Google API Error: {e}")
                    except Exception as e:
                        st.error(f"Failed: {e}")
                        st.info("Check: 1. Sheet name 2. Shared with service account 3. Header row")

# ==================== FOOTER ====================
st.markdown("<div style='text-align:center; padding:2rem; color:#666;'>Zephyr • 8 Models • Daily Newsletter • Live News • VADER Sentiment</div>", unsafe_allow_html=True)