# !pip install xgboost
# !pip install fbprophet
# !pip install keras
# !pip install tensorflow
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from prophet import Prophet
from keras.models import Sequential
from keras.layers import LSTM, Dense
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings("ignore")


def Train_Test(data, column):
    train = data[column][:int(0.8 * len(data))]
    test = data[column][int(0.8 * len(data)) :]
    return train, test


# Plot function
def plot_results(train, test, predictions, title):
    plt.figure(figsize=(10, 6))
    plt.plot(train.index, train, label="Train")
    plt.plot(test.index, test, label="Test")
    plt.plot(test.index, predictions, label="Predictions")
    plt.title(title)
    plt.legend()
    plt.show()


# ARIMA model
def arima_forecast(train, test):
    model = ARIMA(train, order=(5, 1, 0))
    model_fit = model.fit()
    predictions = model_fit.forecast(steps=len(test))
    return predictions


def plot_arima_forecast(train, test):
    predictions = arima_forecast(train, test)
    title = "ARIMA Forecast"
    plot_results(train, test, predictions, title)


# SARIMAX model
def sarimax_forecast(train, test):
    model = SARIMAX(train, order=(5, 1, 0), seasonal_order=(1, 1, 1, 12))
    model_fit = model.fit()
    predictions = model_fit.forecast(steps=len(test))
    return predictions


def plot_sarimax_forecast(train, test):
    predictions = sarimax_forecast(train, test)
    title = "SARIMAX Forecast"
    plot_results(train, test, predictions, title)


# Prophet model
def prophet_forecast(data, column):
    prophet_data = data.reset_index().rename(columns={"date": "ds", column: "y"})
    train,test = Train_Test(data,column)
    model = Prophet()
    model.fit(prophet_data[:int(0.8 * len(prophet_data))])
    future = model.make_future_dataframe(periods=len(data) - int(0.8 * len(data)))
    forecast = model.predict(future)
    plt.figure(figsize=(10, 6))
    model.plot(forecast)
    plt.title(f"Prophet Forecast - {column}")
    plt.show()
    return forecast['yhat'].iloc[-len(test):]

# Tree-based models
def tree_based_forecast(train, test, model_name="RandomForest"):
    train_x, test_x = np.arange(len(train)).reshape(-1, 1), np.arange(len(train), len(train) + len(test)).reshape(-1, 1)
    if model_name == "RandomForest":
        model = RandomForestRegressor()
    elif model_name == "DecisionTree":
        model = DecisionTreeRegressor()
    elif model_name == "XGBoost":
        model = XGBRegressor()
    model.fit(train_x, train)
    predictions = model.predict(test_x)
    return predictions


def plot_tree_based_forecast(train, test, model_name="RandomForest"):
    predictions = tree_based_forecast(train, test, model_name)
    title = f"{model_name} Forecast"
    plot_results(train, test, predictions, title)


# LSTM model
def lstm_forecast(train, test):
    train_array = np.array(train).reshape(-1, 1)
    test_array = np.array(test).reshape(-1, 1)

    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(1, 1)))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    x_train = np.array([train_array[i:i + 1] for i in range(len(train_array) - 1)])
    y_train = train_array[1:]
    model.fit(x_train, y_train, epochs=10, batch_size=32, verbose=0)

    predictions = []
    input_seq = train_array[-1].reshape(1, 1, 1)
    for _ in range(len(test)):
        next_val = model.predict(input_seq)[0, 0]
        predictions.append(next_val)
        input_seq = np.append(input_seq[:, :, 1:], [[[next_val]]], axis=2)

    return predictions


def plot_lstm_forecast(train, test):
    predictions = lstm_forecast(train, test)
    title = "LSTM Forecast"
    plot_results(train, test, predictions, title)


def plot_last_day_forecast(test, predictions, model_name):
    last_day_index = test.index[-1]
    plt.figure(figsize=(10, 6))
    plt.plot(test.index, test, label="True Values")
    plt.plot(test.index, predictions, label=f"{model_name} Predictions")
    plt.title(f"{model_name} Forecast vs True Values for Last Day")
    plt.legend()
    plt.show()
