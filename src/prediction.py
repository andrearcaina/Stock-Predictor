import pandas as pd
import numpy as np
import requests as req
from src.model import *
from src.api import key

def predict(symbol):
    API, SYM, CUR, LIM = key, "ETH", "USD", 2000

    resp = req.get(f"https://min-api.cryptocompare.com/data/v2/histoday?fsym={SYM}&tsym={CUR}&limit={LIM}&api_key={API}")
    
    data = resp.json()
    df = pd.DataFrame(data["Data"]["Data"])
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df.set_index("time", inplace=True)
    df = df[['close']]
    df = df.rename(columns={'close': 'price'})

    data, scale = scale_data(df)

    look_back = 60
    X, Y = create_dataset(data, look_back=look_back)

    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = Y[:train_size], Y[train_size:]

    model = train_model(X_train, y_train)

    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    y_pred = make_predictions(model, X_test, scale)

    plot_results(y_test, y_pred, symbol)

    return y_pred.tolist()
