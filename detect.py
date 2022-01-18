import sys
import matplotlib.pyplot as plt
import keras
import random
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler


# Parsing input parameters
if len(sys.argv) >= 3 and sys.argv[1] == "-d":
    dataset = sys.argv[2]
else:
    print("Usage: detect.py -d <dataset> -n <number of time series selected> -mae <error value as double> -m <model directory>")
    sys.exit()
n = 1
if len(sys.argv) >= 5 and sys.argv[3] == "-n":
    n = int(sys.argv[4])
mae = 0.65
if len(sys.argv) >= 7 and sys.argv[5] == "-mae":
    mae = float(sys.argv[6])
saved_model = "models/model_detect"
if len(sys.argv) >= 9 and sys.argv[7] == "-m":
    saved_model = sys.argv[8]

dataframe = pd.read_csv(dataset,'\t',header=None).iloc[:,:]

# Array to mark timeseries already printed
selected_timeseries = []
for i in range(0,n):
    # Repeat until a timeseries that has not been printed is selected
    while True:
        ts = random.randint(0,len(dataframe)-1)
        if ts not in selected_timeseries:
            # Mark timeseries as selected and use in this iteration
            selected_timeseries.append(ts)
            break

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


    model = keras.models.load_model(saved_model)


    X_train_pred = model.predict(X_train)
    train_mae_loss = np.mean(np.abs(X_train_pred - X_train), axis=1)

    THRESHOLD = mae


    X_test_pred = model.predict(X_test)
    test_mae_loss = np.mean(np.abs(X_test_pred - X_test), axis=1)


    test_score_df = pd.DataFrame(index=test[TIME_STEPS:].index)
    test_score_df['loss'] = test_mae_loss
    test_score_df['threshold'] = THRESHOLD
    test_score_df['anomaly'] = test_score_df.loss > test_score_df.threshold
    test_score_df['close'] = test[TIME_STEPS:].close

    anomalies = test_score_df[test_score_df.anomaly == True]

    print("anomalies: ", len(anomalies))

    plt.plot(
        test[TIME_STEPS:].index,
        test_close_backup[TIME_STEPS:],
        label='close price'
    )

    if (len(anomalies)):
        sns.scatterplot(
            anomalies.index,
            scaler.inverse_transform(anomalies)[:,3],
            color=sns.color_palette()[3],
            s=52,
            label='anomaly'
        )

    plt.xticks(rotation=25)
    plt.title(stock + ' Stock Anomaly Detection')
    plt.legend()
    plt.show()