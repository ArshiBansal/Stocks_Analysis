import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import yfinance as yf  # kept for .info only (optional)
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Models
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from xgboost import XGBRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, GRU
from tensorflow.keras.callbacks import EarlyStopping

# Stooq data source
import pandas_datareader.data as web
from urllib.error import HTTPError
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)
import requests
from pandas_datareader._utils import RemoteDataError

# ==================== PAGE CONFIG & THEME ====================
st.set_page_config(page_title="Magnificent 7+ AI Forecaster Pro", page_icon="🤖", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .main-header {font-size: 3.5rem; font-weight: 800; text-align: center;
        background: linear-gradient(90deg, #00ff88, #00d4ff);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;}
    .sub-header {font-size: 1.8rem; color: #00ff88; font-weight: 600; margin: 1.5rem 0;}
    .stTabs [data-baseweb="tab"] {background-color: #16213e; border-radius: 10px 10px 0 0; padding: 14px 28px; color: #e0e0e0;}
    .stTabs [aria-selected="true"] {background-color: #00ff881a !important; border-bottom: 4px solid #00ff88; color: #00ff88;}
    .css-1d391kg {background-color: #0f0f1e;}
</style>
""", unsafe_allow_html=True)

# ==================== STOCKS & SIDEBAR ====================
stocks = {
    "Magnificent 7": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"],
    "Top Tech Giants": ["AMD", "NFLX", "ADBE", "ORCL", "CRM", "INTC", "PYPL", "AVGO", "QCOM", "TXN"]
}

st.sidebar.markdown("<h2 style='color:#00ff88;'>AI Forecaster Pro</h2><p style='color:#888;'>6 Models + Deep Learning</p>", unsafe_allow_html=True)
category = st.sidebar.selectbox("Stock Group", list(stocks.keys()))
ticker = st.sidebar.selectbox("Select Stock", stocks[category])
forecast_days = st.sidebar.slider("Forecast Days", 10, 180, 60, 10)
lookback_days = st.sidebar.slider("Lookback Window", 30, 180, 90, 10)

# ==================== CLEAR CACHE & REFRESH ====================
if st.sidebar.button("Clear Cache & Full Refresh", use_container_width=True):
    st.cache_data.clear()
    keys_to_remove = ['last_data_fetch', 'models_ts_ml', 'backtest_ts_ml', 'backtest_dl', 'lstm_model', 'gru_model']
    for key in keys_to_remove:
        if key in st.session_state:
            del st.session_state[key]
    logger.info("Cache cleared for ticker %s", ticker)
    st.rerun()

# ==================== DATA LOADER – Stooq (most reliable for Streamlit Cloud) ====================
@st.cache_data(ttl=3600, show_spinner="Fetching latest data from Stooq...")
def load_data(symbol):
    df = pd.DataFrame()
    # Try pandas_datareader first
    try:
        stooq_symbol = f"{symbol}.US"
        end = datetime.now()
        start = end - timedelta(days=5*365 + 150)
        df = web.DataReader(stooq_symbol, 'stooq', start, end)
        if not df.empty:
            df = df.reset_index()
            df = df.rename(columns={'Date':'Date', 'Open':'Open', 'High':'High', 'Low':'Low', 'Close':'Close', 'Volume':'Volume'})
            df = df.sort_values('Date').reset_index(drop=True)
            logger.info("Data loaded via pandas-datareader for %s", symbol)
            return df
except NotImplementedError as e:
    logger.error(f"Stooq data source is not supported: {e}")
    st.warning("Primary data source is unavailable. Trying backup data source...")

except (requests.exceptions.RequestException, RemoteDataError, ValueError, KeyError) as e:
    logger.warning("pandas-datareader failed for %s: %s. Trying CSV fallback.", symbol, e)

except Exception as e:
    logger.exception(f"Unexpected error while fetching data for {symbol}: {e}")

    # Fallback: direct CSV from Stooq
    try:
        url = f"https://stooq.com/q/d/l/?s={symbol.lower()}&i=d"
        temp = pd.read_csv(url)
        if not temp.empty:
            temp['<DATE>'] = pd.to_datetime(temp['<DATE>'], format='%Y%m%d')
            temp = temp.rename(columns={
                '<DATE>':'Date', '<OPEN>':'Open', '<HIGH>':'High',
                '<LOW>':'Low', '<CLOSE>':'Close', '<VOL>':'Volume'
            })
            df = temp[['Date','Open','High','Low','Close','Volume']].sort_values('Date').reset_index(drop=True)
            logger.info("Data loaded via CSV fallback for %s", symbol)
            return df
except HTTPError as e:
    logger.error(f"Backup CSV URL returned an HTTP error: {e}")

except (pd.errors.EmptyDataError, pd.errors.ParserError, OSError) as e:
    logger.error("CSV fallback also failed for %s: %s", symbol, e)

except Exception as e:
    logger.exception(f"Unexpected error while fetching backup CSV for {symbol}: {e}")

logger.error("All data sources exhausted for %s", symbol)
return pd.DataFrame()
    logger.error("All data sources exhausted for %s", symbol)
    return pd.DataFrame()

# Load data
df = load_data(ticker)

if df.empty:
    st.error("Unable to fetch historical stock data. Please verify the ticker symbol or try again later.")
    logger.warning("Empty DataFrame for %s – stopping app.", ticker)
    st.stop()

# Remove timezone if any
if df['Date'].dt.tz is not None:
    df['Date'] = df['Date'].dt.tz_localize(None)

# Last fetch info
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
        Data from Stooq • refreshed at {fetch_time}
    </p>
</div>
""", unsafe_allow_html=True)

# Technical indicators
df['SMA_50'] = df['Close'].rolling(50).mean()
df['SMA_200'] = df['Close'].rolling(200).mean()
df['Return'] = df['Close'].pct_change()
df['Volatility_30d'] = df['Return'].rolling(30).std() * np.sqrt(252)
df['Cumulative_Return'] = (1 + df['Return']).cumprod() - 1

# Header
st.markdown(f'<h1 class="main-header">{ticker} — AI Forecast Dashboard</h1>', unsafe_allow_html=True)

col1, col2, col3, col4, col5, col6 = st.columns(6)
latest = df.iloc[-1]
change = (latest.Close - df['Close'].iloc[-2]) / df['Close'].iloc[-2] * 100 if len(df) > 1 else 0

with col1: st.metric("Price", f"${latest.Close:.2f}", f"{change:+.2f}%")
with col2: st.metric("Volume", f"{latest.Volume/1e6:.1f}M")
with col3: st.metric("52W High", f"${df['High'].max():.2f}")
with col4: st.metric("52W Low", f"${df['Low'].min():.2f}")
# Optional: try to get market cap / P/E from yfinance (may fail on cloud)
try:
    info = yf.Ticker(ticker).info
    with col5: st.metric("Market Cap", f"${info.get('marketCap',0)/1e12:.2f}T")
    with col6: st.metric("P/E Ratio", f"{info.get('trailingPE','N/A'):.2f}" if info.get('trailingPE') else "N/A")
except (ValueError, KeyError, ConnectionError) as e:
    logger.warning("yfinance info unavailable for %s: %s", ticker, e)

except Exception as e:
    logger.exception(f"Unable to fetch company info for {ticker}: {e}")

with col5:
    st.metric("Market Cap", "N/A")
with col6:
    st.metric("P/E Ratio", "N/A")
    with col5: st.metric("Market Cap", "N/A")
    with col6: st.metric("P/E Ratio", "N/A")

st.markdown("---")

# ==================== TABS ====================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Overview", "Raw Data", "Time Series + ML", "Neural Networks", "Performance Metrics", "AI Forecast"
])

# ==================== TAB 1: OVERVIEW ====================
with tab1:
    st.markdown('<div class="sub-header">Comprehensive Stock Overview</div>', unsafe_allow_html=True)
    
    fig1 = make_subplots(specs=[[{"secondary_y": True}]])
    fig1.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name="Close Price", line=dict(color="#00ff88", width=3)))
    fig1.add_trace(go.Scatter(x=df['Date'], y=df['SMA_50'], name="SMA 50", line=dict(dash="dash", color="#ffa500")))
    fig1.add_trace(go.Scatter(x=df['Date'], y=df['SMA_200'], name="SMA 200", line=dict(dash="dot", color="#ff00ff")))
    fig1.add_trace(go.Bar(x=df['Date'], y=df['Volume'], name="Volume", opacity=0.25, marker_color="#333333"), secondary_y=True)
    fig1.update_layout(title="Price Trend with Moving Averages & Volume", template="plotly_dark", height=500)
    fig1.update_yaxes(title_text="Price ($)", secondary_y=False)
    fig1.update_yaxes(title_text="Volume", secondary_y=True)
    st.plotly_chart(fig1, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        fig2 = go.Figure()
        fig2.add_trace(go.Histogram(x=df['Return'].dropna()*100, nbinsx=70, name="Daily Returns", marker_color="#00d4ff", opacity=0.8))
        fig2.update_layout(title="Daily Returns Distribution (%)", template="plotly_dark", height=400, xaxis_title="Return (%)", yaxis_title="Frequency")
        st.plotly_chart(fig2, use_container_width=True)
    
    with col_b:
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=df['Date'], y=df['Volatility_30d'], name="30D Volatility", line=dict(color="#ff00aa", width=2)))
        fig3.update_layout(title="30-Day Rolling Volatility (Annualized)", template="plotly_dark", height=400)
        st.plotly_chart(fig3, use_container_width=True)

    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(x=df['Date'], y=df['Cumulative_Return']*100, name="Cumulative Return %", line=dict(color="rgba(0, 255, 0, 0.9)", width=3)))
    fig4.add_hline(y=0, line_dash="dash", line_color="gray")
    fig4.update_layout(title="Cumulative Returns (Buy & Hold Strategy)", template="plotly_dark", height=450, yaxis_title="Total Return (%)")
    st.plotly_chart(fig4, use_container_width=True)

# ==================== TAB 2: RAW DATA ====================
with tab2:
    st.markdown('<div class="sub-header">Historical Data</div>', unsafe_allow_html=True)
    st.dataframe(df.tail(200), use_container_width=True)
    st.download_button("Download Historical Data", df.to_csv(index=False).encode(), f"{ticker}_history.csv", "text/csv")

# ==================== TAB 3: TIME SERIES + ML ====================
with tab3:
    st.markdown('<div class="sub-header">Train 6 Models + Backtest Results</div>', unsafe_allow_html=True)
    
    if st.button("Train All 6 Models", type="primary", use_container_width=True):
        with st.spinner("Training models..."):
            models = {}
            backtest_data = {}
            
            # ARIMA
            with st.expander("1. ARIMA"):
                try:
                    series = df['Close']
                    model = ARIMA(series, order=(5,1,0))
                    fitted = model.fit()
                    steps = len(series) // 5
                    pred = fitted.forecast(steps=steps)
                    actual = series.iloc[-steps:]
                    backtest_data['ARIMA'] = (df['Date'].iloc[-steps:].values, actual.values, pred.values)
                    models['ARIMA'] = fitted
                    st.success("ARIMA trained")
                    logger.info("ARIMA model trained for %s", ticker)
                # Keep broad exception: a single model failure should not stop the remaining models
                except Exception as e:
                    logger.exception("ARIMA training failed for %s: %s", ticker, e)
                    st.error("ARIMA model could not be trained. The time series may be too short or contain irregularities. Skipping this model.")

            # Prophet
            with st.expander("2. Prophet"):
                try:
                    p_df = df[['Date', 'Close']].copy()
                    p_df.columns = ['ds', 'y']
                    p_df['ds'] = pd.to_datetime(p_df['ds']).dt.tz_localize(None)
                    m = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
                    m.add_seasonality(name='monthly', period=30.5, fourier_order=5)
                    m.fit(p_df)
                    future = m.make_future_dataframe(periods=0)
                    fc = m.predict(future)
                    pred = fc['yhat'].values[-len(df)//5:]
                    actual = df['Close'].iloc[-len(pred):]
                    backtest_data['Prophet'] = (df['Date'].iloc[-len(pred):].values, actual.values, pred)
                    models['Prophet'] = m
                    st.success("Prophet trained")
                    logger.info("Prophet model trained for %s", ticker)
                except Exception as e:
                    logger.exception("Prophet training failed for %s: %s", ticker, e)
                    st.error("Prophet model could not be trained. This may be due to insufficient data or date parsing issues. Skipping this model.")

            # Lagged features for ML models
            lagged = pd.DataFrame({f'lag_{i}': df['Close'].shift(i) for i in range(1,21)})
            lagged['target'] = df['Close']
            lagged.dropna(inplace=True)
            split = int(0.8 * len(lagged))
            X_train = lagged.drop('target', axis=1).iloc[:split]
            X_test  = lagged.drop('target', axis=1).iloc[split:]
            y_train = lagged['target'].iloc[:split]
            y_test  = lagged['target'].iloc[split:]
            test_dates = df['Date'].iloc[-len(y_test):].values

            # Random Forest
            with st.expander("3. Random Forest"):
                try:
                    rf = RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=42)
                    rf.fit(X_train, y_train)
                    pred = rf.predict(X_test)
                    backtest_data['Random Forest'] = (test_dates, y_test.values, pred)
                    models['Random Forest'] = rf
                    st.success("RF trained")
                    logger.info("Random Forest trained for %s", ticker)
                except Exception as e:
                    logger.exception("Random Forest training failed for %s: %s", ticker, e)
                    st.error("Random Forest model could not be trained. Skipping this model.")

            # XGBoost
            with st.expander("4. XGBoost"):
                try:
                    xgb = XGBRegressor(n_estimators=500, learning_rate=0.05, n_jobs=-1, random_state=42)
                    xgb.fit(X_train, y_train)
                    pred = xgb.predict(X_test)
                    backtest_data['XGBoost'] = (test_dates, y_test.values, pred)
                    models['XGBoost'] = xgb
                    st.success("XGBoost trained")
                    logger.info("XGBoost trained for %s", ticker)
                except Exception as e:
                    logger.exception("XGBoost training failed for %s: %s", ticker, e)
                    st.error("XGBoost model could not be trained. Skipping this model.")

            # Unsupervised
            with st.expander("5-6. K-Means & Isolation Forest"):
                try:
                    features = df[['Close', 'Volume', 'Return', 'Volatility_30d']].fillna(0)
                    models['KMeans'] = KMeans(n_clusters=4, random_state=42, n_init=10).fit(features)
                    models['IsolationForest'] = IsolationForest(contamination=0.02, random_state=42).fit(features)
                    st.success("Unsupervised trained")
                    logger.info("KMeans and IsolationForest trained for %s", ticker)
                except Exception as e:
                    logger.exception("Unsupervised model training failed for %s: %s", ticker, e)
                    st.error("K-Means / Isolation Forest models could not be trained. Skipping.")

            st.session_state.models_ts_ml = models
            st.session_state.backtest_ts_ml = backtest_data
            n_trained = len(models)
            st.success(f"{n_trained} model(s) trained!")
            logger.info("Training complete for %s: %d/6 models succeeded.", ticker, n_trained)

    if st.session_state.get("backtest_ts_ml"):
        model_name = st.selectbox("Select Model", [""] + list(st.session_state.backtest_ts_ml.keys()), key="tsml")
        if model_name:
            dates_raw, actual, pred = st.session_state.backtest_ts_ml[model_name]
            dates = pd.to_datetime(dates_raw).strftime("%Y-%m-%d")
            results = pd.DataFrame({
                "Date": dates,
                "Actual": np.round(actual, 2),
                "Predicted": np.round(pred, 2),
                "Error": np.round(pred - actual, 2)
            })
            st.markdown(f"#### {model_name} — Backtest")
            st.dataframe(results.style.format({"Actual": "${:.2f}", "Predicted": "${:.2f}", "Error": "${:.2f}"}), use_container_width=True)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=dates, y=actual, name="Actual", line=dict(color="white")))
            fig.add_trace(go.Scatter(x=dates, y=pred, name="Predicted", line=dict(color="#00ff88", dash="dot")))
            fig.update_layout(template="plotly_dark", height=450, title=f"{model_name} Backtest")
            st.plotly_chart(fig, use_container_width=True)

# ==================== TAB 4: NEURAL NETWORKS ====================
with tab4:
    st.markdown('<div class="sub-header">Train LSTM & GRU + Backtest</div>', unsafe_allow_html=True)
    
    if st.button("Train LSTM + GRU", type="primary", use_container_width=True):
        with st.spinner("Training deep models..."):
            scaler = MinMaxScaler()
            scaled = scaler.fit_transform(df['Close'].values.reshape(-1, 1))

            def create_dataset(data, steps):
                X, y = [], []
                for i in range(steps, len(data)):
                    X.append(data[i-steps:i])
                    y.append(data[i])
                return np.array(X), np.array(y)

            try:
                X, y = create_dataset(scaled, lookback_days)
                X = X.reshape((X.shape[0], X.shape[1], 1))
                split = int(0.8 * len(X))
                test_dates = df['Date'].iloc[split + lookback_days:].values
            # Keep broad exception: dataset creation can fail for many reasons (short data, bad lookback, etc.)
            except Exception as e:
                logger.exception("Failed to create deep learning dataset for %s: %s", ticker, e)
                st.error("Unable to prepare deep learning data. The lookback window may be too large for the available data. Try reducing the lookback window in the sidebar.")
                st.stop()

            backtest_dl = {}

            # LSTM
            with st.expander("LSTM"):
                try:
                    lstm_model = Sequential([
                        LSTM(100, return_sequences=True, input_shape=(lookback_days,1)),
                        Dropout(0.3),
                        LSTM(100),
                        Dropout(0.3),
                        Dense(1)
                    ])
                    lstm_model.compile(optimizer='adam', loss='mse')
                    lstm_model.fit(X[:split], y[:split], epochs=50, batch_size=32, verbose=0, callbacks=[EarlyStopping(patience=10)])
                    pred_scaled = lstm_model.predict(X[split:], verbose=0)
                    pred = scaler.inverse_transform(pred_scaled).flatten()
                    actual = scaler.inverse_transform(y[split:]).flatten()
                    backtest_dl['LSTM'] = (test_dates, actual, pred)
                    st.session_state.lstm_model = (lstm_model, scaler)
                    st.success("LSTM trained")
                    logger.info("LSTM trained for %s", ticker)
                except Exception as e:
                    logger.exception("LSTM training failed for %s: %s", ticker, e)
                    st.error("LSTM model could not be trained. This may be due to insufficient data or numerical instability. Skipping LSTM.")

            # GRU
            with st.expander("GRU"):
                try:
                    gru_model = Sequential([
                        GRU(100, return_sequences=True, input_shape=(lookback_days,1)),
                        Dropout(0.3),
                        GRU(100),
                        Dropout(0.3),
                        Dense(1)
                    ])
                    gru_model.compile(optimizer='adam', loss='mse')
                    gru_model.fit(X[:split], y[:split], epochs=50, batch_size=32, verbose=0, callbacks=[EarlyStopping(patience=10)])
                    pred_scaled = gru_model.predict(X[split:], verbose=0)
                    pred = scaler.inverse_transform(pred_scaled).flatten()
                    actual = scaler.inverse_transform(y[split:]).flatten()
                    backtest_dl['GRU'] = (test_dates, actual, pred)
                    st.session_state.gru_model = (gru_model, scaler)
                    st.success("GRU trained")
                    logger.info("GRU trained for %s", ticker)
                except Exception as e:
                    logger.exception("GRU training failed for %s: %s", ticker, e)
                    st.error("GRU model could not be trained. This may be due to insufficient data or numerical instability. Skipping GRU.")

            st.session_state.backtest_dl = backtest_dl
            n_dl = len(backtest_dl)
            st.success(f"{n_dl} deep learning model(s) trained!")
            logger.info("Deep learning training complete for %s: %d/2 models succeeded.", ticker, n_dl)

    if st.session_state.get("backtest_dl"):
        dl_model = st.selectbox("Select Model", [""] + list(st.session_state.backtest_dl.keys()), key="dl")
        if dl_model:
            dates_raw, actual, pred = st.session_state.backtest_dl[dl_model]
            dates = pd.to_datetime(dates_raw).strftime("%Y-%m-%d")
            results = pd.DataFrame({
                "Date": dates,
                "Actual": np.round(actual, 2),
                "Predicted": np.round(pred, 2),
                "Error": np.round(pred - actual, 2)
            })
            st.markdown(f"#### {dl_model} — Backtest")
            st.dataframe(results.style.format({"Actual": "${:.2f}", "Predicted": "${:.2f}", "Error": "${:.2f}"}), use_container_width=True)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=dates, y=actual, name="Actual", line=dict(color="white")))
            fig.add_trace(go.Scatter(x=dates, y=pred, name="Predicted", line=dict(color="#ff00aa", dash="dot")))
            fig.update_layout(template="plotly_dark", height=450, title=f"{dl_model} Backtest")
            st.plotly_chart(fig, use_container_width=True)

# ==================== TAB 5: PERFORMANCE METRICS ====================
with tab5:
    st.markdown('<div class="sub-header">Model Performance — 5 Key Metrics</div>', unsafe_allow_html=True)
    metrics_list = []

    def add_metrics(name, actual, pred):
        try:
            mae = mean_absolute_error(actual, pred)
            rmse = np.sqrt(mean_squared_error(actual, pred))
            mape = np.mean(np.abs((actual - pred) / (actual + 1e-8))) * 100
            r2 = r2_score(actual, pred)
            directional = np.mean(np.sign(np.diff(actual)) == np.sign(np.diff(pred))) * 100 if len(actual) > 1 else 0
            metrics_list.append({
                "Model": name,
                "MAE": mae,
                "RMSE": rmse,
                "MAPE (%)": mape,
                "R² Score": r2,
                "Direction Accuracy (%)": directional
            })
        except Exception as e:
            logger.warning("Could not compute metrics for %s: %s", name, e)

    if st.session_state.get("backtest_ts_ml"):
        for name, (_, actual, pred) in st.session_state.backtest_ts_ml.items():
            add_metrics(name, actual, pred)

    if st.session_state.get("backtest_dl"):
        for name, (_, actual, pred) in st.session_state.backtest_dl.items():
            add_metrics(name, actual, pred)

    if metrics_list:
        metrics_df = pd.DataFrame(metrics_list)
        metrics_df = metrics_df.round({"MAE": 2, "RMSE": 2, "MAPE (%)": 2, "R² Score": 3, "Direction Accuracy (%)": 1})
        metrics_df = metrics_df.sort_values("MAE").reset_index(drop=True)
        metrics_df.index += 1
        st.dataframe(metrics_df.style
            .format({"MAE": "${:.2f}", "RMSE": "${:.2f}", "MAPE (%)": "{:.2f}%", "R² Score": "{:.3f}", "Direction Accuracy (%)": "{:.1f}%"})
            .highlight_min(subset=["MAE", "RMSE", "MAPE (%)"], color='#00ff8820')
            .highlight_max(subset=["R² Score", "Direction Accuracy (%)"], color='#00ff8820'),
            use_container_width=True)
        best = metrics_df.iloc[0]["Model"]
        st.success(f"Best Model: **{best}**")
    else:
        st.info("Train models to see performance comparison")

# ==================== TAB 6: AI FORECAST ====================
with tab6:
    st.markdown(f'<div class="sub-header">AI Forecast — Next {forecast_days} Days</div>', unsafe_allow_html=True)
    
    if not (st.session_state.get("models_ts_ml") or st.session_state.get("lstm_model")):
        st.warning("Please train models first!")
    else:
        future_dates = pd.bdate_range(start=df['Date'].iloc[-1] + timedelta(days=1), periods=forecast_days)
        forecast_data = {"Date": future_dates.strftime("%Y-%m-%d")}
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name="Historical", line=dict(color="white", width=3)))

        if 'Prophet' in st.session_state.get("models_ts_ml", {}):
            try:
                future = st.session_state.models_ts_ml['Prophet'].make_future_dataframe(periods=forecast_days, freq='B')
                fc = st.session_state.models_ts_ml['Prophet'].predict(future)
                prophet_pred = fc['yhat'][-forecast_days:].values
                forecast_data["Prophet"] = np.round(prophet_pred, 2)
                fig.add_trace(go.Scatter(x=future_dates, y=prophet_pred, name="Prophet", line=dict(color="#00d4ff", width=3)))
            except Exception as e:
                logger.exception("Prophet forecast failed for %s: %s", ticker, e)
                st.warning("Prophet forecast could not be generated. Skipping Prophet.")

        if st.session_state.get("lstm_model"):
            try:
                model, scaler = st.session_state.lstm_model
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
                fig.add_trace(go.Scatter(x=future_dates, y=lstm_pred, name="LSTM", line=dict(color="#ff00aa", dash="dot", width=3)))
            except Exception as e:
                logger.exception("LSTM forecast failed for %s: %s", ticker, e)
                st.warning("LSTM forecast could not be generated. Skipping LSTM.")

        fig.update_layout(template="plotly_dark", height=600, title=f"{ticker} — Multi-Model Forecast")
        st.plotly_chart(fig, use_container_width=True)

        forecast_df = pd.DataFrame(forecast_data)
        st.markdown("#### Forecast Values")
        st.dataframe(forecast_df.style.format({"Prophet": "${:.2f}", "LSTM": "${:.2f}"}), use_container_width=True)

        csv = forecast_df.to_csv(index=False).encode()
        st.download_button(
            label="Download Forecast CSV",
            data=csv,
            file_name=f"{ticker}_forecast_{forecast_days}d.csv",
            mime="text/csv",
            use_container_width=True
        )
        st.success("Forecast generated!")

st.markdown("<div style='text-align:center; padding:2rem; color:#666;'>"
            "AI Stock Forecaster • Stooq Data • Multiple Models • 2026</div>",
            unsafe_allow_html=True)