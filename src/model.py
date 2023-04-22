import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM

def scale_data(df):
    scale = MinMaxScaler(feature_range=(0, 1))
    data = scale.fit_transform(df.values)
    return data, scale

def create_dataset(data, look_back=1):
    X, Y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:(i + look_back), 0])
        Y.append(data[i + look_back, 0])
    return np.array(X), np.array(Y)

def train_model(X_train, y_train):
    seq_model = Sequential()
    seq_model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    seq_model.add(Dropout(0.2))
    seq_model.add(LSTM(units=50, return_sequences=True))
    seq_model.add(Dropout(0.2))
    seq_model.add(LSTM(units=50))
    seq_model.add(Dropout(0.2))
    seq_model.add(Dense(units=1))
    seq_model.compile(optimizer='adam', loss='mean_squared_error')

    seq_model.fit(X_train, y_train, epochs=5, batch_size=32)

    return seq_model

def make_predictions(model, X_test, scale):
    y_pred = model.predict(X_test)
    y_pred = scale.inverse_transform(y_pred)
    return y_pred

def plot_results(actual, predicted, stock_symbol):
    plt.figure(figsize=(12,6))
    plt.plot(actual, label='Actual')
    plt.plot(predicted, label='Predicted')
    plt.title(f'{stock_symbol} Price Prediction (LSTM)')
    plt.xlabel('Time')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.show()

def predict(df, symbol):
    # Scale the data
    data, scale = scale_data(df)

    # Create the training and testing datasets
    look_back = 60
    X, Y = create_dataset(data, look_back=look_back)

    # Split the data into training and testing sets
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = Y[:train_size], Y[train_size:]

    # Train the model
    model = train_model(X_train, y_train)

    # Make predictions
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    y_pred = make_predictions(model, X_test, scale)

    # Plot the results
    plot_results(y_test, y_pred, symbol)

    # Return the predictions
    return y_pred.tolist()
