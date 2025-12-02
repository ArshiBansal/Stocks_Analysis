<h1 align="center">📈 AI Stock Market Forecasting & Analytics</h1>

---

Problem Statement ❗

Financial markets are highly volatile and influenced by numerous unpredictable factors.
Traditional forecasting methods struggle to capture nonlinear patterns, regime changes, and sudden anomalies in stock prices.

The challenge:

Develop an AI-driven system that can analyze historical stock market data, detect anomalies, identify market regimes, and generate accurate multi-model forecasts for major technology stocks.

The goal of this project is to build a complete stock forecasting ecosystem that leverages machine learning, deep learning, and statistical modeling to help analysts and retail investors make data-driven decisions.

---

# Overview 📊

This project provides a full AI-powered stock forecasting platform, combining:
  - A Streamlit web application for interactive forecasting
  - A robust data analysis pipeline for feature engineering, ML model training, and statistical analysis

Stocks covered include:

  i. AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA and,
  
  ii. Major tech giants like AMD, NFLX, INTC, QCOM, AVGO, CRM, ORCL, PYPL, etc.

The system offers forecasting, anomaly detection, market regime clustering, and deep insights through charts, metrics, and multi-model evaluation.

---

# Features ⚡

### 📉 1. Market Analysis & Visualization

  - Price trends with SMA-50, SMA-200
  - Candlestick-style line charts
  - Volume overlay
  - Daily returns distribution
  - Rolling 30-day volatility
  - Cumulative returns (Buy & Hold)

### 🔮 2. AI Forecasting (10–180 days)

  - Prophet forecasting
  - LSTM autoregressive predictions
  - Future predictions shown via Plotly
  - Export forecast CSV

### 🤖 3. Machine Learning Models (6 Models)

  - Available in the web app:
      a. ARIMA

      b. Prophet

      c. Random Forest Regressor

      d. XGBoost Regressor

      e. K-Means Clustering

      f. Isolation Forest (Anomaly Detection)

  - Available in the offline pipeline:

      a. Linear Regression, Ridge, Lasso

      b. LightGBM

      c. MLP Neural Network

      d. Wide & Deep Network

      e. SARIMA

      f. PCA & Gaussian Mixture Models

### 🧠 4. Deep Learning Models

  - LSTM (Sequential)

  - GRU

  - Keras Dense Network

  - Wide & Deep Neural Network

### 🚨 5. Anomaly Detection

  - Isolation Forest detects flash crashes, extreme returns, or irregular volume spikes

  - Heatmaps of anomalous months

### 🔍 6. Market Regime Detection

  - KMeans clusters identify: Bull market, Bear market, Sideways consolidation, High-volatility regimes.

### 📈 7. Backtesting & Model Evaluation

  - Metrics include: MAE, RMSE, MAPE, R² Score, Direction Accuracy, Automatically identifies Best Model.

### 🗂 8. Full Data Engineering Pipeline

  - In the offline script (data_analysis_pipeline.py):

    - Cleaning missing OHLCV

    - Feature engineering

    - Lag features

    - Momentum indicators

    - Volatility features

    - Market position metrics

    - Exporting cleaned dataset

### 🎛️ 9. Interactive Filters

  Choose: Stock group, Ticker, Lookback window, Forecast horizon
 
---

# Technologies Used 🛠️

### 🐍 Python Libraries
  - Streamlit
  - Pandas, NumPy
  - Plotly
  - Scikit-Learn
  - Prophet
  - pmdarima
  - LightGBM
  - XGBoost
  - TensorFlow/Keras
  - yfinance
  - Statsmodels
  - PCA, GMM
  - stumpy (pattern detection)

### 📊 Visualization
  - Plotly (live charts)
  - Matplotlib, Seaborn (offline pipeline)

### 🧠 Machine Learning & Deep Learning
  - Regression models
  - Gradient boosting
  - Neural networks
  - LSTM / GRU
  - Clustering
  - Isolation Forest
    
---

# Installation 🧩

```bash
git clone https://github.com/YourUsername/AI-Stock-Forecaster.git
cd AI-Stock-Forecaster
pip install -r requirements.txt
```

---

# Data Requirements 📂

- Anime Dataset: major-tech-stock-2019-2024.csv from kaggle and yfinance python library

- Key Columns: `date`, `open`, `high`, `low`, `close`, `adj close`, `volume`, `ticker`

### 🧠 Feature Engineering Includes:
- Creating Date Columns: `Year`, `Month`, `Day`, `Day of Week`
- Creating Columns: `Daily return`, `MA (for 7, 30 & 90 days)`,`Volatility`,`Lag close`,`Momentum`,`Price Position`,`Volume_MA`,`Volume Ratio`

---

# Usage Guide 🚀

 **Run App**:  
  ```bash
streamlit run analysis.py
  ```
 ### 📂 Explore Tabs in the App
  - Overview — Price, volume, volatility, returns
  - Raw Data — Download historical data
  - Time Series + ML — Train 6 models
  - Neural Networks — Train LSTM/GRU
  - Performance Metrics — Compare all models
  - AI Forecast — Predict next 10–180 days

### 🧪 Train Models
  - Click the training buttons inside each tab:
  - “Train All 6 Models”
  - “Train LSTM + GRU”
  - Results are cached for speed.

---

# Limitations ⚠️
  - Forecast accuracy decreases during high-volatility periods.
  - Deep learning models require ~50+ days of lookback to work effectively.
  - Prophet seasonal components may oversmooth short-term volatility.
  - Isolation Forest may mark false anomalies depending on contamination rate.
  - Does not include macroeconomic indicators (future improvement).

---

# Future Improvements 🔮
  - Add Temporal Fusion Transformer (TFT)
  - Add DeepAR or N-BEATS for long-horizon forecasting
  - Integrate news sentiment/LLM embeddings
  - Deploy app to AWS/Streamlit Cloud
  - Add portfolio risk metrics (VaR, CVaR, Sharpe optimizer)
  - Implement Reinforcement Learning trading agent

---

# Notes 📝
  - Cached models stored automatically using Streamlit session state
  - Forecasting uses business days only
  - UI is fully responsive with neon-dark theme

---
- 🔐 Implement user login to save preferences & personalized dashboards.
  
