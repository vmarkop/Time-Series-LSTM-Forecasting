import sys
import matplotlib.pyplot as plt
import keras
import random
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler


# Parsing input parameters
if len(sys.argv) >= 5 and sys.argv[1] == "-d" and sys.argv[3] == "-o":
    dataset = sys.argv[2]
    output = sys.argv[4]
else:
    print("Usage: python detect_trainall.py -d <dataset> -o <output>")
    sys.exit()

dataframe = pd.read_csv(dataset,'\t',header=None).iloc[:,:]

model = keras.Sequential()

selected_timeseries = []
for ts in range(0, len(dataframe)):

    stock = dataframe.iloc[ts][0]
    timeseries = dataframe.iloc[ts,1:]
    df1 = pd.DataFrame(timeseries)
    df  = pd.DataFrame()

    dates = range(1,3650)
    dates = pd.DataFrame(dates)
    df['close'] = df1
    df['index'] = dates
    df.set_index('index')

    train_size = int(len(df) * 0.95)
    test_size = len(df) - train_size
    train, test = df.iloc[0:train_size], df.iloc[train_size:len(df)]
    print(train.shape, test.shape)

    scaler = StandardScaler()
    scaler = scaler.fit(train[['close']])

    print(test[['close']].shape)

    test_close_backup = test[['close']]

    train['close'] = scaler.transform(train[['close']])
    test['close'] = scaler.transform(test[['close']])

    print(test[['close']].shape)

    def create_dataset(X, y, time_steps=1):
        Xs, ys = [], [] 

        for i in range(len(X) - time_steps):
            v = X.iloc[i:(i + time_steps)].values
            Xs.append(v)
            ys.append(y.iloc[i + time_steps])

        return np.array(Xs), np.array(ys)


    TIME_STEPS = 30

    # reshape to [samples, time_steps, n_features]

    print(train.shape)

    X_train, y_train = create_dataset(
        train[['close']],
        train.close,
        TIME_STEPS
    )

    X_test, y_test = create_dataset(
        test[['close']],
        test.close,
        TIME_STEPS
    )

    print(X_train.shape)

    # Initialize model on first iteration
    if ts == 0:
        model.add(keras.layers.LSTM(
            units=64,
            input_shape=(X_train.shape[1], X_train.shape[2])
        ))

        model.add(keras.layers.Dropout(rate=0.2))
        model.add(keras.layers.RepeatVector(n=X_train.shape[1]))
        model.add(keras.layers.LSTM(units=64, return_sequences=True))
        model.add(keras.layers.Dropout(rate=0.2))
        model.add(
        keras.layers.TimeDistributed(
            keras.layers.Dense(units=X_train.shape[2])
        )
        )
        model.compile(loss='mae', optimizer='adam')


    history = model.fit(
        X_train, y_train,
        epochs=10,
        batch_size=32,
        validation_split=0.1,
        shuffle=False
    )

    # Backup every 50 timeseries
    if ts % 50 == 0:
        model.save(output)

model.save(output)