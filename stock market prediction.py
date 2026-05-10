import yfinance as yf
from gnews import GNews
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import schedule
import time
import datetime
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, LSTM, GRU, Dense, Dropout, Bidirectional
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, RMSprop, Nadam
from tensorflow.keras.regularizers import l2
from sklearn.ensemble import StackingRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
import logging
import os
from retrying import retry

log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    filename="nifty50_prediction.log",
    level=getattr(logging, log_level, logging.INFO),
    format="%(asctime)s - %(levelname)s - %(message)s"
)

data = pd.read_csv(r"D:\stock prediction\NIFTY_50_Historical_PR.csv")
data['Date'] = pd.to_datetime(data['Date'], dayfirst=True)
data.set_index('Date', inplace=True)

def feature_engineering(df):
    """Apply feature engineering to the given DataFrame."""
    df['MA_20'] = df['Close'].rolling(window=20).mean()
    df['MA_50'] = df['Close'].rolling(window=50).mean()
    df['Bollinger_Upper'] = df['MA_20'] + 2 * df['Close'].rolling(window=20).std()
    df['Bollinger_Lower'] = df['MA_20'] - 2 * df['Close'].rolling(window=20).std()
    df['RSI'] = 100 - (100 / (1 + df['Close'].pct_change().rolling(14).apply(
        lambda x: (x[x > 0].sum() / -x[x < 0].sum()) if -x[x < 0].sum() != 0 else 0)))
    df['MACD'] = df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()
    df['Stochastic_Oscillator'] = (df['Close'] - df['Low'].rolling(window=14).min()) / (
        df['High'].rolling(window=14).max() - df['Low'].rolling(window=14).min())
    df['ATR'] = (df['High'] - df['Low']).rolling(window=14).mean()
    df['Lag1'] = df['Close'].shift(1)
    df['Lag2'] = df['Close'].shift(2)
    df['Rolling_Mean_30'] = df['Close'].rolling(window=30).mean()
    df['Rolling_Std_30'] = df['Close'].rolling(window=30).std()
    return df.dropna()

data = feature_engineering(data)

scalers = {}
normalized_data = pd.DataFrame(index=data.index)
for column in ['Close', 'MA_20', 'MA_50', 'Bollinger_Upper', 'Bollinger_Lower', 'RSI', 'MACD', 'Stochastic_Oscillator', 'ATR', 'Lag1', 'Lag2', 'Rolling_Mean_30', 'Rolling_Std_30']:
    scalers[column] = MinMaxScaler(feature_range=(0, 1))
    normalized_data[column] = scalers[column].fit_transform(data[[column]])

def create_sequences(features, target, seq_length):
    x, y = [], []
    for i in range(seq_length, len(features)):
        x.append(features[i-seq_length:i])
        y.append(target[i])
    return np.array(x), np.array(y)

seq_length = 60
x_data, y_data = create_sequences(
    normalized_data.drop(columns=['Close']).values,
    normalized_data['Close'].values,
    seq_length
)

tscv = TimeSeriesSplit(n_splits=5)
train_indices, test_indices = list(tscv.split(x_data))[-1]
x_train, x_test = x_data[train_indices], x_data[test_indices]
y_train, y_test = y_data[train_indices], y_data[test_indices]

def build_gru_model(seq_length, num_features, optimizer='adam', dropout_rate=0.2, l2_reg=0.01):
    inputs = Input(shape=(seq_length, num_features))
    x = GRU(64, return_sequences=True, kernel_regularizer=l2(l2_reg))(inputs)
    x = Dropout(dropout_rate)(x)
    x = GRU(32, kernel_regularizer=l2(l2_reg))(x)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(1)(x)
    model = Model(inputs, outputs)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model

param_grid = {
    'optimizer': ['adam', 'rmsprop', 'nadam'],
    'dropout_rate': [0.2, 0.3],
    'l2_reg': [0.01, 0.001],
    'batch_size': [32, 64],
    'epochs': [50, 100]
}

def grid_search(x_train, y_train):
    best_model = None
    best_score = float('inf')
    
    for optimizer in param_grid['optimizer']:
        for dropout_rate in param_grid['dropout_rate']:
            for l2_reg in param_grid['l2_reg']:
                for batch_size in param_grid['batch_size']:
                    for epochs in param_grid['epochs']:
                        model = build_gru_model(x_train.shape[1], x_train.shape[2], optimizer, dropout_rate, l2_reg)
                        model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=0)
                        predictions = model.predict(x_test)
                        mse = mean_squared_error(y_test, predictions)
                        if mse < best_score:
                            best_score = mse
                            best_model = model
    return best_model

best_gru_model = grid_search(x_train, y_train)

xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6)
stacked_model = StackingRegressor(
    estimators=[('gru', best_gru_model), ('xgb', xgb_model)],
    final_estimator=LinearRegression()
)

stacked_model.fit(x_train.reshape(x_train.shape[0], -1), y_train)

predictions = stacked_model.predict(x_test.reshape(x_test.shape[0], -1))
predictions_scaled = scalers['Close'].inverse_transform(predictions.reshape(-1, 1))
test_actual = scalers['Close'].inverse_transform(y_test.reshape(-1, 1))

mse = mean_squared_error(test_actual, predictions_scaled)
mae = mean_absolute_error(test_actual, predictions_scaled)
r2 = r2_score(test_actual, predictions_scaled)

print(f"Mean Squared Error: {mse}")
print(f"Mean Absolute Error: {mae}")
print(f"R-Squared: {r2}")

results = pd.DataFrame(index=data.index[-len(test_actual):])
results['Actual'] = test_actual.flatten()
results['Predicted'] = predictions_scaled.flatten()
results.to_csv(r"D:\stock prediction\NIFTY_50_Historical_PR_Results.csv")

plt.figure(figsize=(16, 8))
plt.plot(results['Actual'], label='Actual Price')
plt.plot(results['Predicted'], label='Predicted Price')
plt.title("Nifty 50 Close Price Prediction (Stacking Ensemble Model)")
plt.legend()
plt.show()

gru_model = best_gru_model
xgb_model = XGBRegressor()
xgb_model.load_model("xgb_model.json")
stacked_model = stacked_model 
scalers = scalers  

news = GNews(language='en', country='IN', max_results=10)
analyzer = SentimentIntensityAnalyzer()

@retry(stop_max_attempt_number=3, wait_fixed=2000)
def fetch_sentiment():
    articles = news.get_news("Nifty 50 stock market")
    scores = [analyzer.polarity_scores(article['title'])['compound'] for article in articles]
    if not scores:
        raise ValueError("No sentiment scores fetched.")
    return np.mean(scores)

@retry(stop_max_attempt_number=3, wait_fixed=2000)
def fetch_market_data():
    df = yf.download("^NSEI", period="7d", interval="5m", progress=False)
    if df.empty:
        raise ValueError("Market data fetch returned empty.")
    df = df.dropna()
    return feature_engineering(df)

def prepare_input(latest_df, sentiment_score):
    features = ['MA_20', 'MA_50', 'Bollinger_Upper', 'Bollinger_Lower', 'RSI', 'MACD',
                'Stochastic_Oscillator', 'ATR', 'Lag1', 'Lag2', 'Rolling_Mean_30', 'Rolling_Std_30']
    latest_df = latest_df[-60:]
    X = pd.DataFrame(index=latest_df.index)

    for col in features:
        X[col] = scalers[col].transform(latest_df[[col]])
    X['Sentiment'] = sentiment_score
    return X.values.reshape(1, 60, -1)

def fetch_realtime_news_sentiment():
    """Fetch real-time news sentiment for Nifty 50."""
    try:
        articles = news.get_news("Nifty 50 stock market")
        if not articles:
            logging.warning("No news articles fetched for sentiment analysis.")
            return 0.0 

        sentiment_scores = []
        for article in articles:
            title = article.get('title', '')
            description = article.get('description', '')
            content = f"{title} {description}"
            sentiment = analyzer.polarity_scores(content)['compound']
            sentiment_scores.append(sentiment)

        avg_sentiment = np.mean(sentiment_scores)
        logging.info(f"Real-time news sentiment score: {avg_sentiment:.4f}")
        return avg_sentiment
    except Exception as e:
        logging.error(f"Error fetching real-time news sentiment: {e}")
        return 0.0 

def classify_sentiment(score, threshold=0.05):
    """Classify sentiment as positive, neutral, or negative based on the score."""
    if score > threshold:
        return "Positive"
    elif score < -threshold:
        return "Negative"
    else:
        return "Neutral"

def realtime_predict():
    try:
        sentiment = fetch_realtime_news_sentiment()
        sentiment_class = classify_sentiment(sentiment)
        logging.info(f"Fetched real-time sentiment score: {sentiment:.4f} ({sentiment_class})")
        
        df = fetch_market_data()
        logging.info("Fetched market data successfully.")
        
        input_data = prepare_input(df, sentiment)
        logging.info("Prepared input data for prediction.")
        
        stacked_input = input_data.reshape(1, -1)
        prediction = stacked_model.predict(stacked_input)[0]
        prediction_real = scalers['Close'].inverse_transform([[prediction]])[0][0]

        logging.info(f"Predicted Close Price: {prediction_real:.2f}")
        print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] 📈 Predicted Close: {prediction_real:.2f} | 📰 Sentiment: {sentiment:.4f} ({sentiment_class})")
    except ValueError as ve:
        logging.error(f"ValueError encountered: {ve}")
        print(f"[ERROR] ValueError: {ve}")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        print(f"[ERROR] {e}")

def cleanup():
    """Perform cleanup before shutdown."""
    try:
        logging.info("Performing cleanup before shutdown.")
        global gru_model, xgb_model, stacked_model
        gru_model = None
        xgb_model = None
        stacked_model = None
        logging.info("Cleared models from memory.")

        global scalers
        scalers.clear()
        logging.info("Cleared scalers.")

        logging.info("Cleanup completed successfully.")
        print("Cleanup completed. Exiting...")
    except Exception as e:
        logging.error(f"Error during cleanup: {e}")
        print(f"[ERROR] Cleanup failed: {e}")

schedule.every(5).minutes.do(realtime_predict)

logging.info("✅ Real-time Nifty 50 Prediction Started (updates every 5 minutes)...")
print("✅ Real-time Nifty 50 Prediction Started (updates every 5 minutes)...")
realtime_predict()

while True:
    try:
        schedule.run_pending()
        time.sleep(1)
    except KeyboardInterrupt:
        logging.info("Real-time prediction stopped by user.")
        cleanup()
        break
    except Exception as e:
        logging.error(f"Unexpected error in main loop: {e}")
        cleanup()
        break
