# %% [markdown]
# # 1. Importing Libraries

# %%
# ====================== 1. CORE & DATA HANDLING ======================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# %%
# ====================== 2. TIME SERIES CLASSICAL ======================
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from pmdarima import auto_arima
from prophet import Prophet

# %%
# ====================== 3. SUPERVISED ML (scikit-learn) ======================
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, classification_report,mean_absolute_percentage_error,explained_variance_score,max_error

# %%
# ====================== 4. UNSUPERVISED ML ======================
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
import umap

# %%
# Autoencoder (TensorFlow example — you can switch to PyTorch if preferred)
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense, Dropout,Input, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings('ignore')
from sklearn.neural_network import MLPRegressor

# %%
# ====================== 6. NEURAL NETWORKS / DEEP LEARNING ======================
# PyTorch Forecasting (TFT + others)
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer, Baseline
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss 

# PyTorch Lightning (for custom models if needed)
import pytorch_lightning as pl

# GluonTS (alternative DeepAR implementation)
from gluonts.torch.model.deepar import DeepAREstimator   

# ====================== 7. PATTERN DETECTION (Bonus – highly recommended) ======================
import stumpy

# %% [markdown]
# # 2. Data Loading and Pipelining (Cleaning)

# %%
# ====================== 1. LOAD DATA ======================
df = pd.read_csv('major-tech-stock-2019-2024.csv')
print(f"Raw data shape: {df.shape}")
print("Columns:", df.columns.tolist())
print("\nSample:")
print(df.head(), "\n")
print(df.tail())

# %%
# ====================== 2. INITIAL CLEANING ======================

# Standardize column names
df.columns = [col.strip().replace(' ', '_').replace('Adj_Close', 'Adj Close') for col in df.columns]

# Set index and sort
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(['Ticker', 'Date']).reset_index(drop=True)

# Drop completely duplicate rows
initial_rows = len(df)
df.drop_duplicates(subset=['Date', 'Ticker'], keep='first', inplace=True)
print(f"Removed {initial_rows - len(df):,} full duplicate rows")


# %%
# Drop rows with missing Date or Ticker
df.dropna(subset=['Date', 'Ticker'], inplace=True)

# Fill or drop missing values intelligently
print(f"\nMissing values before cleaning:")
print(df.isnull().sum())

# %%
# Forward fill typical missing OHLCV (happens on holidays/weekends per ticker)
df[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']] = \
    df.groupby('Ticker')[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']].ffill()

# If still missing (e.g. first rows), backfill
df[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']] = \
    df.groupby('Ticker')[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']].bfill()

# Volume = 0 or NaN → set to 0 (some datasets mark holidays as 0)
df['Volume'] = df['Volume'].fillna(0)

print(f"\nMissing values after cleaning:")
print(df.isnull().sum())

# %%
# ====================== 3. ENHANCED FEATURE ENGINEERING ======================

df['Year']        = df['Date'].dt.year
df['Month']       = df['Date'].dt.month
df['Day']         = df['Date'].dt.day
df['DayOfWeek']   = df['Date'].dt.dayofweek          # Monday=0, Sunday=6
df['Is_Month_End'] = df['Date'].dt.is_month_end.astype(int)
df['Is_Month_Start'] = df['Date'].dt.is_month_start.astype(int)

# Daily return from Open to Close (intraday momentum)
df['Daily_Return'] = (df['Close'] - df['Open']) / df['Open']

# Moving averages
df['MA7']  = df.groupby('Ticker')['Close'].transform(lambda x: x.rolling(window=7).mean())
df['MA30'] = df.groupby('Ticker')['Close'].transform(lambda x: x.rolling(window=30).mean())
df['MA90'] = df.groupby('Ticker')['Close'].transform(lambda x: x.rolling(window=90).mean())
# 30-day realized volatility on intraday returns
df['Volatility_30d'] = df.groupby('Ticker')['Daily_Return'].transform(
    lambda x: x.rolling(window=30, min_periods=10).std()
)

# Lagged prices (very powerful for ML models)
df['Lag_Close_1'] = df.groupby('Ticker')['Close'].shift(1)
df['Lag_Close_2'] = df.groupby('Ticker')['Close'].shift(2)
df['Lag_Close_3'] = df.groupby('Ticker')['Close'].shift(3)
df['Lag_Close_5'] = df.groupby('Ticker')['Close'].shift(5)

# Price momentum
df['Momentum_5d']  = df.groupby('Ticker')['Close'].transform(lambda x: x.pct_change(5))
df['Momentum_10d'] = df.groupby('Ticker')['Close'].transform(lambda x: x.pct_change(10))

# Price position within recent range
df['Price_Position_20d'] = (
    df['Close'] - df.groupby('Ticker')['Close'].transform(lambda x: x.rolling(20).min())
) / (
    df.groupby('Ticker')['Close'].transform(lambda x: x.rolling(20).max()) - 
    df.groupby('Ticker')['Close'].transform(lambda x: x.rolling(20).min()) + 1e-8
)

# Volume features
df['Volume_MA20'] = df.groupby('Ticker')['Volume'].transform(lambda x: x.rolling(20).mean())
df['Volume_Ratio'] = df['Volume'] / (df['Volume_MA20'] + 1e-6)

# High-Low range & typical price
df['HL_Range'] = df['High'] - df['Low']
df['Typical_Price'] = (df['High'] + df['Low'] + df['Close']) / 3

# Overnight gap (useful for next-day prediction)
df['Overnight_Gap'] = df.groupby('Ticker')['Open'].pct_change()

# Target variable (what most models will predict)
# Option 1: Next-day close price direction (classification)
df['Target_Direction'] = (df.groupby('Ticker')['Close'].shift(-1) > df['Close']).astype(int)

# %%
# ====================== FINAL CLEANING AFTER FEATURES ======================
print(f"\nRows before dropping NaNs: {len(df)}")
df_clean = df.dropna().reset_index(drop=True)
print(f"Rows after dropping NaNs: {len(df_clean)}")
print(f"Final shape: {df_clean.shape}")
print(f"Date range: {df_clean['Date'].min().date()} to {df_clean['Date'].max().date()}")
df_clean.to_csv('tech_stocks_final_clean_engineered.csv', index=False)

# %% [markdown]
# # 3. Cross-Stock COMPARISON ANALYSIS

# %%
df = df_clean.copy()
print(df.head())
print("Columns:", df.columns.tolist())
print(f"Tickers in dataset        : {sorted(df['Ticker'].unique())}")
print(f"Total trading days per stock:")
print(df.groupby('Ticker')['Date'].nunique().sort_values(ascending=False))
print()

# %%
# ──────────────────────────────────────
# 1. TOTAL RETURN OVER THE PERIOD
# ──────────────────────────────────────
print("1. CUMULATIVE TOTAL RETURN (Buy & Hold)")
total_returns = df.groupby('Ticker').apply(
    lambda x: (x['Adj Close'].iloc[-1] / x['Adj Close'].iloc[0] - 1) * 100
).round(2).sort_values(ascending=False)

print(total_returns.to_string())
print(f"→ Best performer : {total_returns.idxmax()} (+{total_returns.max():.1f}%)")
print(f"→ Worst performer: {total_returns.idxmin()} ({total_returns.min():+.1f}%)")
print()

# %%
# ──────────────────────────────────────
# 2. ANNUALIZED RETURN & VOLATILITY (Sharpe proxy)
# ──────────────────────────────────────
print("2. ANNUALIZED RETURN & RISK")
stats = df.groupby('Ticker').agg(
    Annualized_Return_pct = ('Daily_Return', lambda x: (1 + x.mean())**252 - 1),
    Annualized_Volatility = ('Daily_Return', lambda x: x.std() * np.sqrt(252)),
    Sharpe_Ratio          = ('Daily_Return', lambda x: (x.mean() / x.std()) * np.sqrt(252)),
    Max_Drawdown_pct      = ('Adj Close', lambda x: ((x.cummax() - x)/x.cummax()).max())
).round(4) * 100

stats['Annualized_Return_pct'] = stats['Annualized_Return_pct'].round(2)
stats['Max_Drawdown_pct'] = stats['Max_Drawdown_pct'].round(2)
stats = stats.sort_values('Sharpe_Ratio', ascending=False)

print(stats)
print(f"→ Highest Sharpe       : {stats.index[0]} ({stats['Sharpe_Ratio'].iloc[0]:.3f})")
print(f"→ Lowest risk-adjusted: {stats.index[-1]} ({stats['Sharpe_Ratio'].iloc[-1]:.3f})")
print()

# %%
# ──────────────────────────────────────
# 3. CORRELATION MATRIX (Daily Returns)
# ──────────────────────────────────────
print("3. DAILY RETURN CORRELATION MATRIX")
corr_matrix = df.pivot(index='Date', columns='Ticker', values='Daily_Return').corr()
print(corr_matrix.round(3))

print(f"→ Highest correlation : {corr_matrix.stack().idxmax()} = {corr_matrix.stack().max():.3f}")
print(f"→ Lowest correlation  : {corr_matrix.stack().idxmin()} = {corr_matrix.stack().min():.3f}")
print()

# %%
plt.figure(figsize=(4,3))
sns.heatmap(corr_matrix, annot=True, cmap='Blues', annot_kws={"color": "Black"})
plt.title('Ticker Correlation Matrix')
plt.savefig('correlation_heatmap.png')
plt.title('Ticker Correlation Matrix', color='white')
plt.xticks(color='white')
plt.yticks(color='white')
plt.show()

# %%
# ──────────────────────────────────────
# 4. TRADING ACTIVITY COMPARISON
# ──────────────────────────────────────
print("4. AVERAGE DAILY VOLUME & VOLATILITY RANK")
volume_vol = df.groupby('Ticker').agg(
    Avg_Daily_Volume_M = ('Volume', 'mean'),
    Volatility_30d_Avg = ('Volatility_30d', 'mean'),
    Pct_Days_Above_5pct_Move = ('Daily_Return', lambda x: (x.abs() > 0.05).mean())
).round(6)

volume_vol['Avg_Daily_Volume_M'] /= 1e6
volume_vol['Pct_Days_Above_5pct_Move'] *= 100
volume_vol = volume_vol.round(3).sort_values('Volatility_30d_Avg', ascending=False)

print(volume_vol)
print(f"→ Most volatile stock : {volume_vol.index[0]}")
print(f"→ Highest trading volume: {volume_vol['Avg_Daily_Volume_M'].idxmax()}")
print()

# %% [markdown]
# # 4. Self-Comparison Over Time

# %%
yearly_stats = df.groupby(['Ticker', 'Year']).agg({
    'Daily_Return': ['mean', 'std'],
    'Close': ['mean']
}).reset_index()
print("\n=== YEARLY STATS PER TICKER (SELF OVER TIME) ===")
print(yearly_stats)

# %% [markdown]
# # 5. CLASSICAL TIME SERIES MODELING

# %%
tickers = df['Ticker'].unique()
results_summary = []

# %%
# ────────────────────── FUNCTION 1: ADF TEST ──────────────────────
def adf_test(series, name="Series"):
    result = adfuller(series, regression='c')
    print(f"  ADF Test ({name})")
    print(f"    ADF Statistic : {result[0]:.6f}")
    print(f"    p-value       : {result[1]:.6f}")
    print(f"    Critical 1%   : {result[4]['1%']:.3f} | 5%: {result[4]['5%']:.3f} | 10%: {result[4]['10%']:.3f}")
    return result[1] < 0.05, result[1]

# %%
# ────────────────────── FUNCTION 2: SARIMA (auto_arima) ──────────────────────
def fit_sarima(series):
    try:
        model = auto_arima(
            series,
            start_p=0, max_p=3,
            start_q=0, max_q=3,
            start_P=0, max_P=2,
            start_Q=0, max_Q=2,
            m=5, seasonal=True,
            stepwise=True,
            suppress_warnings=True,
            error_action='ignore',
            trace=False
        )
        print(f"    Best SARIMA: {model.order} x {model.seasonal_order} (AIC: {model.aic():.2f})")
        return model.aic(), model.order, model.seasonal_order
    except:
        print("    auto_arima failed → skipped")
        return np.inf, None, None

# %%
# ────────────────────── FUNCTION 3: PROPHET ──────────────────────
def fit_prophet(df_prophet):
    m = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode='multiplicative'
    )
    m.add_country_holidays(country_name='US')
    m.fit(df_prophet)
    
    future = m.make_future_dataframe(periods=60)
    forecast = m.predict(future)
    
    # In-sample MAPE
    y_true = df_prophet['y'].values
    y_pred = forecast.iloc[:-60]['yhat'].values
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    print(f"    Prophet In-sample MAPE: {mape:.2f}%")
    return mape

# %%
# ────────────────────── MAIN LOOP ──────────────────────
results = []
for ticker in df_clean['Ticker'].unique():
    print(f"\n{'─'*25} TICKER: {ticker.upper()} {'─'*25}")
    
    # Prepare data
    ts = df_clean[df_clean['Ticker'] == ticker].copy()
    ts = ts.sort_values('Date').set_index('Date')
    
    price = ts['Adj Close']
    # Use proper log returns (Close-to-Close)
    log_returns = np.log(ts['Adj Close'] / ts['Adj Close'].shift(1)).dropna()

    # 1. ADF Tests
    stationary_price, p_price = adf_test(price, "Adj Close Price")
    stationary_ret, p_ret = adf_test(log_returns, "Log Returns")

    # Choose series for SARIMA
    if stationary_price:
        model_series = price
        series_name = "Adj Close (stationary)"
    else:
        model_series = log_returns
        series_name = "Log Returns (stationary)"

    # 2. SARIMA
    sarima_aic, sarima_order, sarima_seasonal = fit_sarima(model_series)

    # 3. Prophet (on price level)
    prophet_df = ts.reset_index()[['Date', 'Adj Close']].rename(columns={'Date': 'ds', 'Adj Close': 'y'})
    prophet_mape = fit_prophet(prophet_df)

    # Store results
    results.append({
        'Ticker'           : ticker,
        'Price_Stationary' : 'Yes' if stationary_price else 'No',
        'Price_p_value'    : round(p_price, 6),
        'Series_Used'      : series_name,
        'SARIMA_AIC'       : round(sarima_aic, 2) if sarima_aic != np.inf else 'Failed',
        'Prophet_MAPE_%'   : round(prophet_mape, 2)
    })

# %% [markdown]
# # 6. SUPERVISED MACHINE LEARNING MODELS

# %%
df_ml = df_clean.copy()
df_ml['Target_Next_Price'] = df_ml.groupby('Ticker')['Adj Close'].shift(-1)
df_ml['Target_Direction'] = (df_ml['Target_Next_Price'] > df_ml['Adj Close']).astype(int)
df_ml = df_ml.dropna(subset=['Target_Next_Price']).reset_index(drop=True)

feature_cols = [
    'Open', 'High', 'Low', 'Volume', 'Year', 'Month', 'Day', 'DayOfWeek',
    'Is_Month_End', 'Is_Month_Start', 'Daily_Return', 'MA7', 'MA30', 'MA90',
    'Volatility_30d', 'Lag_Close_1', 'Lag_Close_2', 'Lag_Close_3', 'Lag_Close_5',
    'Momentum_5d', 'Momentum_10d', 'Price_Position_20d',
    'Volume_MA20', 'Volume_Ratio', 'HL_Range', 'Typical_Price', 'Overnight_Gap'
]

# %%
X_raw = df_ml[feature_cols]
y = df_ml['Target_Next_Price']
dates = df_ml['Date']
tickers = df_ml['Ticker']

# Scale features
scaler = StandardScaler()
X = scaler.fit_transform(X_raw)
X = pd.DataFrame(X, columns=feature_cols)

# Train-test split by time (last 20% as test)
split_idx = int(len(df_ml) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
dates_test = dates.iloc[split_idx:]
tickers_test = tickers.iloc[split_idx:]

print(f"Training samples  : {len(X_train):,}")
print(f"Test samples      : {len(X_test):,}")
print(f"Test date range   : {dates_test.min().date()} → {dates_test.max().date()}")

# %%
# MODEL 1: Linear Regression
model_lr = LinearRegression()
model_lr.fit(X_train, y_train)
pred_lr = model_lr.predict(X_test)

# %%
# 6 Metrics
mae = mean_absolute_error(y_test, pred_lr)
rmse = np.sqrt(mean_squared_error(y_test, pred_lr))
mape = mean_absolute_percentage_error(y_test, pred_lr) * 100
r2 = r2_score(y_test, pred_lr)
evs = explained_variance_score(y_test, pred_lr)
max_err = max_error(y_test, pred_lr)

print(f"MAE                    : {mae:.3f}")
print(f"RMSE                   : {rmse:.3f}")
print(f"MAPE                   : {mape:.2f}%")
print(f"R² Score               : {r2:.4f}")
print(f"Explained Variance     : {evs:.4f}")
print(f"Max Error              : {max_err:.3f}")

# %%
# Sample predictions
print("\nSample Predictions (Last 10 days):")
results_lr = pd.DataFrame({
    'Date': dates_test.reset_index(drop=True),
    'Ticker': tickers_test.reset_index(drop=True),
    'Actual': y_test.reset_index(drop=True),
    'Predicted': pred_lr
})
results_lr['Error'] = results_lr['Actual'] - results_lr['Predicted']
print(results_lr.tail(10).round(2))

# %%
# Plot
plt.figure(figsize=(7, 3))
plt.plot(results_lr['Date'], results_lr['Actual'], label='Actual Price', alpha=0.8)
plt.plot(results_lr['Date'], results_lr['Predicted'], label='Linear Reg Predicted', alpha=0.8)
plt.title('Linear Regression: Actual vs Predicted Next-Day Price', fontsize=14)
plt.xlabel('Date')
plt.ylabel('Adjusted Close Price')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %%
# MODEL 2: Ridge Regression
model_ridge = Ridge(alpha=1.0)
model_ridge.fit(X_train, y_train)
pred_ridge = model_ridge.predict(X_test)

# %%
mae = mean_absolute_error(y_test, pred_ridge)
rmse = np.sqrt(mean_squared_error(y_test, pred_ridge))
mape = mean_absolute_percentage_error(y_test, pred_ridge) * 100
r2 = r2_score(y_test, pred_ridge)
evs = explained_variance_score(y_test, pred_ridge)
max_err = max_error(y_test, pred_ridge)

print(f"MAE                    : {mae:.3f}")
print(f"RMSE                   : {rmse:.3f}")
print(f"MAPE                   : {mape:.2f}%")
print(f"R² Score               : {r2:.4f}")
print(f"Explained Variance     : {evs:.4f}")
print(f"Max Error              : {max_err:.3f}")

# %%
results_ridge = pd.DataFrame({
    'Date': dates_test.reset_index(drop=True),
    'Actual': y_test.reset_index(drop=True),
    'Predicted': pred_ridge
})
results_ridge['Error'] = results_ridge['Actual'] - results_ridge['Predicted']
print("\nSample Predictions:")
print(results_ridge.tail(10).round(2))

# %%
plt.figure(figsize=(7, 3))
plt.plot(results_ridge['Date'], results_ridge['Actual'], label='Actual', alpha=0.8)
plt.plot(results_ridge['Date'], results_ridge['Predicted'], label='Ridge Predicted', color='green', alpha=0.8)
plt.title('Ridge Regression: Actual vs Predicted', fontsize=14)
plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout(); plt.show()

# %%
# MODEL 3: Lasso Regression
model_lasso = Lasso(alpha=0.001, max_iter=20000)
model_lasso.fit(X_train, y_train)
pred_lasso = model_lasso.predict(X_test)

# %%
mae = mean_absolute_error(y_test, pred_lasso)
rmse = np.sqrt(mean_squared_error(y_test, pred_lasso))
mape = mean_absolute_percentage_error(y_test, pred_lasso) * 100
r2 = r2_score(y_test, pred_lasso)
evs = explained_variance_score(y_test, pred_lasso)
max_err = max_error(y_test, pred_lasso)

print(f"MAE                    : {mae:.3f}")
print(f"RMSE                   : {rmse:.3f}")
print(f"MAPE                   : {mape:.2f}%")
print(f"R² Score               : {r2:.4f}")
print(f"Explained Variance     : {evs:.4f}")
print(f"Max Error              : {max_err:.3f}")

# %%
results_lasso = pd.DataFrame({
    'Date': dates_test.reset_index(drop=True), 
    'Actual': y_test.reset_index(drop=True), 
    'Predicted': pred_lasso
})
results_lasso['Error'] = results_lasso['Actual'] - results_lasso['Predicted']
print("\nSample Predictions:")
print(results_lasso.tail(10).round(2))

# %%
plt.figure(figsize=(7, 3))
plt.plot(results_lasso['Date'], results_lasso['Actual'], label='Actual')
plt.plot(results_lasso['Date'], results_lasso['Predicted'], label='Lasso Predicted', color='red')
plt.title('Lasso Regression: Actual vs Predicted'); 
plt.legend(); 
plt.grid(); 
plt.show()

# %%
# MODEL 4: Random Forest Regressor
model_rf = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
model_rf.fit(X_train, y_train)
pred_rf = model_rf.predict(X_test)

# %%
mae = mean_absolute_error(y_test, pred_rf)
rmse = np.sqrt(mean_squared_error(y_test, pred_rf))
mape = mean_absolute_percentage_error(y_test, pred_rf) * 100
r2 = r2_score(y_test, pred_rf)
evs = explained_variance_score(y_test, pred_rf)
max_err = max_error(y_test, pred_rf)

print(f"MAE                    : {mae:.3f}")
print(f"RMSE                   : {rmse:.3f}")
print(f"MAPE                   : {mape:.2f}%")
print(f"R² Score               : {r2:.4f}")
print(f"Explained Variance     : {evs:.4f}")
print(f"Max Error              : {max_err:.3f}")

# %%
results_rf = pd.DataFrame({
    'Date': dates_test.reset_index(drop=True), 
    'Actual': y_test.reset_index(drop=True), 
    'Predicted': pred_rf
})
results_rf['Error'] = results_rf['Actual'] - results_rf['Predicted']
print("\nSample Predictions:")
print(results_rf.tail(10).round(2))

# %%
plt.figure(figsize=(7, 3))
plt.plot(results_rf['Date'], results_rf['Actual'], label='Actual', linewidth=1.5)
plt.plot(results_rf['Date'], results_rf['Predicted'], label='Random Forest Predicted', color='darkorange', alpha=0.9)
plt.title('Random Forest: Best Performing Model', fontsize=16, fontweight='bold')
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout(); plt.show()

# %% [markdown]
# # 7. Gradient Booster MACHINE LEARNING MODELS

# %%
# 5. LIGHTGBM REGRESSOR
# LightGBM Dataset
train_data = lgb.Dataset(X_train, label=y_train)
test_data  = lgb.Dataset(X_test,  label=y_test, reference=train_data)

params = {
    'objective': 'regression',    # We want to predict continuous values
    'metric': 'mae',              # Mean Absolute Error — both optimization and evaluation metric
    'boosting_type': 'gbdt',      # Classic gradient boosting decision trees
    'num_leaves': 31,             # Max leaves per tree (31 is default, good starting point)
    'learning_rate': 0.05,        # Small step size → slower but more accurate learning
    'feature_fraction': 0.9,      # Randomly select 90% of features at each tree (reduces overfitting)
    'bagging_fraction': 0.8,      # Randomly select 80% of data for each tree (like subsample)
    'bagging_freq': 5,            # Perform bagging every 5 iterations
    'verbose': -1,                # Suppress most warnings/info
    'seed': 42                    # For reproducibility
}

# === TRAIN WITH CALLBACK (ONLY WORKING METHOD IN 2025+) ===
model_lgb = lgb.train(
    params,
    train_data,
    num_boost_round=2000,                 # Maximum number of trees (iterations)
    valid_sets=[test_data],               # Datasets to evaluate during training
    valid_names=['valid'],                # Name shown in logs
    callbacks=[
        lgb.early_stopping(stopping_rounds=100, verbose=True),
        lgb.log_evaluation(period=100)
    ]
)

# === PREDICT ===
best_iter = model_lgb.best_iteration
pred_lgb = model_lgb.predict(X_test, num_iteration=best_iter)

# %%
mae  = mean_absolute_error(y_test, pred_lgb)
rmse = np.sqrt(mean_squared_error(y_test, pred_lgb))
mape = mean_absolute_percentage_error(y_test, pred_lgb) * 100
r2   = r2_score(y_test, pred_lgb)
evs  = explained_variance_score(y_test, pred_lgb)
maxe = max_error(y_test, pred_lgb)

print(f"\nLightGBM FINAL RESULTS (Best Iteration: {best_iter}):")
print(f"   MAE                : {mae:.3f}")
print(f"   RMSE               : {rmse:.3f}")
print(f"   MAPE               : {mape:.2f}%")
print(f"   R² Score           : {r2:.4f}")
print(f"   Explained Variance : {evs:.4f}")
print(f"   Max Error          : {maxe:.3f}")

# %%
# === OUTPUT TABLE ===
results = pd.DataFrame({
    'Date': dates_test,
    'Ticker': tickers_test,
    'Actual': y_test.round(2).values,
    'Predicted': np.round(pred_lgb, 2),
    'Error': np.round(y_test.values - pred_lgb, 2)
})
print("\nLast 10 Predictions:")
print(results.tail(10).to_string(index=False))

# %%
# === PLOT ===
plt.figure(figsize=(8, 3.5))
plt.plot(results['Date'], results['Actual'], label='Actual Price', linewidth=2, color='steelblue')
plt.plot(results['Date'], results['Predicted'], label='LightGBM Predicted', linewidth=2, color='darkgreen', alpha=0.9)
plt.title('LightGBM – Best Model So Far (All 5 Tech Stocks)', fontsize=16, fontweight='bold')
plt.xlabel('Date')
plt.ylabel('Adjusted Close Price ($)')
plt.legend(fontsize=12)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# %%
# MODEL 6: XGBoost Regressor
dtrain = xgb.DMatrix(X_train, label=y_train)   # Training data
dtest  = xgb.DMatrix(X_test,  label=y_test)    # Test/validation data (used for early stopping)

params = {
    'objective'        : 'reg:squarederror',   # We want to minimize squared error
    'eval_metric'      : 'mae',                # Monitor Mean Absolute Error during training
    'eta'              : 0.05,                 # Learning rate (smaller = more robust)
    'max_depth'        : 6,                    # Max tree depth (6–8 is good for tabular data)
    'subsample'        : 0.8,                  # Use 80% of rows per tree → reduces overfitting
    'colsample_bytree' : 0.8,                  # Use 80% of features per tree
    'seed'             : 42,                   # For reproducibility
    'verbosity'        : 0                     # 0 = silent, 1 = info, 2 = warning
}

# We allow up to 2000 trees, but stop if no improvement in 100 rounds
model_xgb = xgb.train(
    params=params,
    dtrain=dtrain,
    num_boost_round=2000,                     # Maximum number of trees
    evals=[(dtest, 'valid')],                 # Validation set (named 'valid')
    early_stopping_rounds=100,                # Stop if MAE doesn't improve for 100 rounds
    verbose_eval=100                          # Print progress every 100 rounds
)

# Best iteration is automatically saved
best_iter = model_xgb.best_iteration
print(f"\nXGBoost stopped early at iteration {best_iter} (saved ~{2000 - best_iter} trees)")
pred_xgb = model_xgb.predict(dtest, iteration_range=(0, best_iter + 1))

# %%
mae  = mean_absolute_error(y_test, pred_xgb)
rmse = np.sqrt(mean_squared_error(y_test, pred_xgb))
mape = mean_absolute_percentage_error(y_test, pred_xgb) * 100
r2   = r2_score(y_test, pred_xgb)
evs  = explained_variance_score(y_test, pred_xgb)
maxe = max_error(y_test, pred_xgb)

print(f"\nXGBOOST FINAL RESULTS (Best Iteration: {best_iter}):")
print(f"   MAE                : {mae:.3f}")
print(f"   RMSE               : {rmse:.3f}")
print(f"   MAPE               : {mape:.2f}%")
print(f"   R² Score           : {r2:.4f}")
print(f"   Explained Variance : {evs:.4f}")
print(f"   Max Error          : {maxe:.3f}")

# %%
results_xgb = pd.DataFrame({
    'Date'      : dates_test.reset_index(drop=True),
    'Ticker'    : tickers_test.reset_index(drop=True),
    'Actual'    : np.round(y_test.values, 2),
    'Predicted' : np.round(pred_xgb, 2),
    'Error'     : np.round(y_test.values - pred_xgb, 2)
})

print("\nLast 10 Predictions:")
print(results_xgb.tail(10).to_string(index=False))

# %%
plt.figure(figsize=(8, 3.5 ))
plt.plot(results_xgb['Date'], results_xgb['Actual'], label='Actual Price', linewidth=2, color='steelblue')
plt.plot(results_xgb['Date'], results_xgb['Predicted'], 
         label='XGBoost Predicted', linewidth=2, color='crimson', alpha=0.9)
plt.title('XGBoost Regressor – Next-Day Price Prediction (All 5 Tech Stocks)', 
          fontsize=16, fontweight='bold')
plt.xlabel('Date')
plt.ylabel('Adjusted Close Price ($)')
plt.legend(fontsize=12)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# # 8. UNSUPERVISED MACHINE LEARNING MODELS

# %%
feature_cols_unsup = [
    'Open', 'High', 'Low', 'Volume', 'Daily_Return', 'Volatility_30d',
    'MA7', 'MA30', 'MA90', 'Lag_Close_1', 'Lag_Close_2', 'Lag_Close_3', 'Lag_Close_5',
    'Momentum_5d', 'Momentum_10d', 'Price_Position_20d',
    'Volume_MA20', 'Volume_Ratio', 'HL_Range', 'Typical_Price', 'Overnight_Gap',
    'Year', 'Month', 'DayOfWeek'
]

X_unsup_raw = df_clean[feature_cols_unsup]
X_unsup = pd.DataFrame(StandardScaler().fit_transform(X_unsup_raw), columns=feature_cols_unsup)
print(f"Unsupervised dataset: {X_unsup.shape[0]:,} samples × {X_unsup.shape[1]} features")

# %%
# TYPE 1: CLUSTERING – K-Means (Market Regime Detection)
# Fit K-Means with 4 clusters (typical for market regimes: bull/bear/sideways/high-vol)
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df_clean['KMeans_Regime'] = kmeans.fit_predict(X_unsup)

# Metrics
sil_score = silhouette_score(X_unsup, df_clean['KMeans_Regime'])
print(f"Silhouette Score: {sil_score:.4f} (0.5+ = good separation)")

# %%
# Stacked bar chart: Time spent in each regime per stock
regime_counts = df_clean.groupby(['Ticker', 'KMeans_Regime']).size().unstack(fill_value=0)
regime_counts = regime_counts[sorted(regime_counts.columns)]

plt.figure(figsize=(6, 3.5))
regime_counts.plot(kind='barh', stacked=True, cmap='tab10', ax=plt.gca(), linewidth=1, edgecolor='white')
plt.title('Market Regimes by Stock – Time Spent in Each Regime', fontsize=16, fontweight='bold')
plt.xlabel('Trading Days')
plt.ylabel('Ticker')
plt.legend(title='Regime', bbox_to_anchor=(1.02, 1), loc='upper left')
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.show()

print("\nRegime Duration Table:")
print(regime_counts)

# %%
# TYPE 2: ANOMALY DETECTION – Isolation Forest (Flash Crashes & Spikes)

# Detect anomalies (2% contamination = ~1 extreme day per month)
iso_forest = IsolationForest(contamination=0.02, random_state=42, n_jobs=-1)
df_clean['Is_Anomaly'] = iso_forest.fit_predict(X_unsup)

# Metrics & counts
anomaly_rate = (df_clean['Is_Anomaly'] == -1).mean() * 100
print(f"Anomaly Detection Rate: {anomaly_rate:.2f}% ({len(df_clean[df_clean['Is_Anomaly'] == -1]):,} extreme days detected)")
print(f"False positive rate: {anomaly_rate:.2f}% (adjust contamination to change)")

# Show top anomalies
anomalies = df_clean[df_clean['Is_Anomaly'] == -1].copy()
anomalies = anomalies.sort_values('Daily_Return', key=abs, ascending=False).head(10)

print("\nTop 10 Extreme Days Detected:")
print(anomalies[['Date', 'Ticker', 'Daily_Return', 'Volume_Ratio']].round(3))

# %%
# Calendar heatmap of anomalies
anomaly_heatmap = df_clean[df_clean['Is_Anomaly'] == -1].copy()
anomaly_heatmap['YearMonth'] = anomaly_heatmap['Date'].dt.to_period('M').astype(str)
heatmap_data = anomaly_heatmap.groupby(['YearMonth', 'Ticker']).size().unstack(fill_value=0)

plt.figure(figsize=(10, 6))
sns.heatmap(heatmap_data, cmap="Reds", annot=True, fmt="d", linewidths=0.5, cbar_kws={'label': 'Anomalies'})
plt.title('Anomaly Calendar – Extreme Market Days by Month & Stock', fontsize=16, fontweight='bold')
plt.xlabel('Ticker')
plt.ylabel('Year-Month')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# %%
# TYPE 3: DIMENSIONALITY REDUCTION – PCA (Feature Compression)
# Fit PCA (keep all components for analysis)
pca_model = PCA(random_state=42)
pca_components = pca_model.fit_transform(X_unsup)
explained_var = pca_model.explained_variance_ratio_
cum_var = np.cumsum(explained_var)

# Components needed for 95% variance
n_95 = np.argmax(cum_var >= 0.95) + 1
print(f"Components for 95% variance: {n_95}/{len(explained_var)}")
print(f"First 3 components explain: {cum_var[2]*100:.1f}% of variance")

# %%
# Scree plot with cumulative line
plt.figure(figsize=(6, 6))
plt.bar(range(1, 11), explained_var[:10], alpha=0.7, color='steelblue', label='Individual Variance')
plt.plot(range(1, 11), cum_var[:10], 'ro-', linewidth=2, markersize=6, label='Cumulative')
plt.axhline(y=0.95, color='red', linestyle='--', alpha=0.7, label='95% Threshold')
plt.title('PCA Scree Plot – Market Feature Compression', fontsize=16, fontweight='bold')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# %%
# Top features driving PC1 (market regime indicator)
pc1_loadings = pd.Series(pca_model.components_[0], index=feature_cols_unsup).abs().nlargest(8)
plt.figure(figsize=(5, 5))
pc1_loadings.plot(kind='bar', color='coral', alpha=0.8)
plt.title('Top 8 Features Driving PC1 (Market Regime Factor)', fontsize=14, fontweight='bold')
plt.ylabel('Absolute Loading')
plt.xlabel('Feature')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# %%
# TYPE 4: DENSITY ESTIMATION – Gaussian Mixture Model (Market States)

# Fit GMM (4 components = soft clustering)
gmm = GaussianMixture(n_components=4, covariance_type='full', random_state=42)
df_clean['GMM_State'] = gmm.fit_predict(X_unsup)
probs = pd.DataFrame(gmm.predict_proba(X_unsup),
                     columns=[f'State_{i}' for i in range(4)])
prob_df = pd.concat([df_clean[['Date','Ticker']].reset_index(drop=True), probs], axis=1)

gmm_sil = silhouette_score(X_unsup, df_clean['GMM_State'])
print(f"GMM Silhouette Score: {gmm_sil:.4f}")
print(f"Log Likelihood: {gmm.lower_bound_:.1f}")

# %%
# Daily average probabilities per state (only numeric columns)
daily_avg = prob_df.groupby('Date').mean(numeric_only=True)  # Key fix
overall_mean = daily_avg.mean()  # Now safe: all columns are numeric

# For nicer state names (optional)
state_names = ['State 0', 'State 1', 'State 2', 'State 3']

# Your 4 original colors
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

# Vertical bar chart
plt.figure(figsize=(5, 5))
bars = plt.bar(state_names, overall_mean.values, 
               color=colors, edgecolor='black', linewidth=1.2, alpha=0.85)

# Add value labels on top of each bar
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{height:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.title('Average Daily Probability of Each Market State (2019–2023)\nGaussian Mixture Model (4 Components)', 
          fontsize=14, fontweight='bold', pad=20)
plt.xlabel('Market State', fontsize=12)
plt.ylabel('Mean Probability', fontsize=12)
plt.ylim(0, 1)
plt.grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.show()


# %%
# Show state transitions per ticker
print("\nMarket State Distribution per Stock:")
state_dist = df_clean.groupby('Ticker')['GMM_State'].value_counts(normalize=True).unstack(fill_value=0).round(3)
print(state_dist)

# %% [markdown]
# # 9. Neural Network 

# %%
# MODEL 1: MLP Regressor (scikit-learn) – Fast & Clean
mlp = MLPRegressor(
    hidden_layer_sizes=(100, 50, 25),
    activation='relu',
    solver='adam',
    alpha=0.001,
    learning_rate='adaptive',
    max_iter=1000,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=20
)

mlp.fit(X_train, y_train)
pred_mlp = mlp.predict(X_test)

# %%
# Metrics
mae = mean_absolute_error(y_test, pred_mlp)
rmse = np.sqrt(mean_squared_error(y_test, pred_mlp))
mape = mean_absolute_percentage_error(y_test, pred_mlp) * 100
r2 = r2_score(y_test, pred_mlp)

print(f"MLP Results → MAE: {mae:.3f} | RMSE: {rmse:.3f} | MAPE: {mape:.2f}% | R²: {r2:.4f}")
print(f"Converged in {mlp.n_iter_} iterations")

# %%
# Plot
results_mlp = pd.DataFrame({'Date': dates_test, 'Actual': y_test, 'Predicted': pred_mlp})
plt.figure(figsize=(10, 4))
plt.plot(results_mlp['Date'], results_mlp['Actual'], label='Actual Price', linewidth=2)
plt.plot(results_mlp['Date'], results_mlp['Predicted'], label='MLP Predicted', color='red', alpha=0.8)
plt.title('MLP Regressor (scikit-learn) – Next-Day Price Prediction', fontsize=16, fontweight='bold')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.legend(); 
plt.grid(alpha=0.3); 
plt.tight_layout(); 
plt.show()

# %%
# MODEL 2: Simple Keras Neural Network (2 Hidden Layers)
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1)  # Output: next-day price
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
model.summary()

# %%
early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=300,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

# Predict
pred_keras = model.predict(X_test).flatten()

# %%
# Metrics
mae = mean_absolute_error(y_test, pred_keras)
rmse = np.sqrt(mean_squared_error(y_test, pred_keras))
mape = mean_absolute_percentage_error(y_test, pred_keras) * 100
r2 = r2_score(y_test, pred_keras)

print(f"\nKeras NN → MAE: {mae:.3f} | RMSE: {rmse:.3f} | MAPE: {mape:.2f}% | R²: {r2:.4f}")
print(f"Best epoch: {len(history.history['loss'])}")

# %%
# Plot training + prediction
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

ax1.plot(history.history['loss'], label='Training Loss')
ax1.plot(history.history['val_loss'], label='Validation Loss')
ax1.set_title('Keras Training History'); ax1.legend(); ax1.grid(alpha=0.3)

results_keras = pd.DataFrame({'Date': dates_test, 'Actual': y_test, 'Predicted': pred_keras})
ax2.plot(results_keras['Date'], results_keras['Actual'], label='Actual')
ax2.plot(results_keras['Date'], results_keras['Predicted'], label='Keras Predicted', color='purple')
ax2.set_title('Keras Neural Network – Prediction'); ax2.legend(); ax2.grid(alpha=0.3)

plt.tight_layout(); plt.show()

# %%
# MODEL 3: Wide & Deep Neural Network (Keras Functional API)
# Wide path (linear)
input_layer = Input(shape=(X_train.shape[1],))
wide = Dense(64, activation='linear')(input_layer)

# Deep path
deep = Dense(128, activation='relu')(input_layer)
deep = Dropout(0.3)(deep)
deep = Dense(64, activation='relu')(deep)
deep = Dropout(0.3)(deep)
deep = Dense(32, activation='relu')(deep)

# %%
# Combine wide + deep
combined = Concatenate()([wide, deep])
output = Dense(1)(combined)

wide_deep = Model(inputs=input_layer, outputs=output)
wide_deep.compile(optimizer=Adam(0.0005), loss='mse', metrics=['mae'])

wide_deep.fit(X_train, y_train, validation_split=0.2, epochs=200, batch_size=64,
              callbacks=[EarlyStopping(patience=15, restore_best_weights=True)], verbose=1)

pred_wd = wide_deep.predict(X_test).flatten()

# %%
mae = mean_absolute_error(y_test, pred_wd)
rmse = np.sqrt(mean_squared_error(y_test, pred_wd))
r2 = r2_score(y_test, pred_wd)

print(f"\nWide & Deep → MAE: {mae:.3f} | RMSE: {rmse:.3f} | R²: {r2:.4f}")

# %%
plt.figure(figsize=(10, 4))
plt.plot(dates_test, y_test, label='Actual', linewidth=2)
plt.plot(dates_test, pred_wd, label='Wide & Deep Predicted', color='darkorange', linewidth=2)
plt.title('Wide & Deep Neural Network – Best of Both Worlds', fontsize=16, fontweight='bold')
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout(); plt.show()


