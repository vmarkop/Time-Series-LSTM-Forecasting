import sys
# import math
import matplotlib.pyplot as plt
import keras
import random
import pandas as pd
import numpy as np
import seaborn as sns
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.layers import LSTM
# from keras.layers import Dropout
# from keras.layers import *
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.metrics import mean_squared_error
# from sklearn.metrics import mean_absolute_error
# from sklearn.model_selection import train_test_split
# from keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler


# Parsing input parameters
if len(sys.argv) >= 3 and sys.argv[1] == "-d":
    dataset = sys.argv[2]
else:
    print("Usage: python forecast.py -d <dataset> -n <number of time series selected> -all [optional]")
    sys.exit()
n = 1
if len(sys.argv) >= 5 and sys.argv[3] == "-n":
    n = int(sys.argv[4])
all = True if len(sys.argv) > 5 and sys.argv[5] == "-all" else False

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




model = keras.Sequential()
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


X_train_pred = model.predict(X_train)
train_mae_loss = np.mean(np.abs(X_train_pred - X_train), axis=1)

THRESHOLD = 0.65


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
    # test_close_backup[TIME_STEPS:],
    test[TIME_STEPS:].close,
    # scaler.inverse_transform(test[TIME_STEPS:].close),
    label='close price'
)

if (len(anomalies)):
    sns.scatterplot(
        anomalies.index,
        anomalies.close,
        color=sns.color_palette()[3],
        s=52,
        label='anomaly'
    )

plt.xticks(rotation=25)
plt.title(stock + ' Stock Anomaly Detection')
plt.legend()
plt.show()