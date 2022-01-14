from keras.layers import Input, Dense, Conv1D, MaxPooling1D, UpSampling1D, BatchNormalization, LSTM, RepeatVector
from keras.models import Model
# from keras.models import model_from_json
# from keras import regularizers
import sys
import random
import datetime
import time
import requests as req
import json
import pandas as pd
import pickle
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import matplotlib.pyplot as plt

# Parameters
window_length = 10
encoding_dim = 3
epochs = 100
test_samples = 2000


#BADUTILS
startdate="01/01/2015"
def mkdate(ts):
    return datetime.datetime.fromtimestamp(
        int(ts)
    ).strftime('%Y-%m-%d')


#Utils
def plot_examples(stock_input, stock_decoded):
    n = 10  
    plt.figure(figsize=(20, 4))
    for i, idx in enumerate(list(np.arange(0, test_samples, 200))):
        # display original
        ax = plt.subplot(2, n, i + 1)
        if i == 0:
            ax.set_ylabel("Input", fontweight=600)
        else:
            ax.get_yaxis().set_visible(False)
        plt.plot(stock_input[idx])
        ax.get_xaxis().set_visible(False)
        

        # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        if i == 0:
            ax.set_ylabel("Output", fontweight=600)
        else:
            ax.get_yaxis().set_visible(False)
        plt.plot(stock_decoded[idx])
        ax.get_xaxis().set_visible(False)

def plot_history(history):
    plt.figure(figsize=(15, 5))
    ax = plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"])
    plt.title("Train loss")
    ax = plt.subplot(1, 2, 2)
    plt.plot(history.history["val_loss"])
    plt.title("Test loss")



# # Datasets retrieval & transformation
# # get data
# start_timestamp = time.mktime(datetime.datetime.strptime(startdate, "%d/%m/%Y").timetuple())
# end_timestamp = int(time.time())
# one_week = 3600*24*7 # s
# one_day = 3600*24 # s
# weeks = list(np.arange(start_timestamp, end_timestamp, one_week))
# days_recorded = (datetime.datetime.fromtimestamp(end_timestamp)-datetime.datetime.fromtimestamp(start_timestamp)).days
# print("days_recorded ",days_recorded)
# data = []
# if not os.path.isfile("data.pickle"):
#     s = req.Session()
#     r = s.get("https://www.coindesk.com/price/")
#     for i in range(1, len(weeks)):
#         start_weekday = mkdate(weeks[i-1])
#         end_weekday = mkdate(weeks[i]-one_day)
#         print(start_weekday, end_weekday)
#         r = s.get("https://api.coindesk.com/charts/data?data=close&startdate={}&enddate={}&exchanges=bpi&dev=1&index=USD".format(start_weekday, end_weekday))
#         ans = json.loads(r.text.replace("cb(", "").replace(");",""))["bpi"]
#         ans.sort(key=lambda x: x[0])
#         for pricepoint in ans:
#             if pricepoint[0]/1000 >= weeks[i-1] and pricepoint[0]/1000 < (weeks[i]-one_day):
#                 data.append([int(pricepoint[0]/1000), pricepoint[1]])
                
#     pickle.dump(data, open("./data.pickle", "wb"))
# else:
#     data = pickle.load(open("./data.pickle", "rb"))

# df = pd.DataFrame(np.array(data)[:,1], columns=['price'])
# df['pct_change'] = df.price.pct_change()
# df['log_ret'] = np.log(df.price) - np.log(df.price.shift(1))

# scaler = MinMaxScaler()
# x_train_nonscaled = np.array([df['log_ret'].values[i-window_length:i].reshape(-1, 1) for i in tqdm(range(window_length+1,len(df['log_ret'])))])
# x_train = np.array([scaler.fit_transform(df['log_ret'].values[i-window_length:i].reshape(-1, 1)) for i in tqdm(range(window_length+1,len(df['log_ret'])))])

# x_test = x_train[-test_samples:]
# x_train = x_train[:-test_samples]

# x_train = x_train.astype('float32')
# x_test = x_test.astype('float32')

# plt.figure(figsize=(20,10))
# plt.plot(np.array(data)[:,1])




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
timeseries = timeseries.values.reshape(-1,1)
df = pd.DataFrame(timeseries, columns=['price'])
df.price = df.price.astype('int')
df['pct_change'] = df.price.pct_change()
df['log_ret'] = np.log(df.price) - np.log(df.price.shift(1))

scaler = MinMaxScaler()
x_train_nonscaled = np.array([df['log_ret'].values[i-window_length:i].reshape(-1, 1) for i in tqdm(range(window_length+1,len(df['log_ret'])))])
x_train = np.array([scaler.fit_transform(df['log_ret'].values[i-window_length:i].reshape(-1, 1)) for i in tqdm(range(window_length+1,len(df['log_ret'])))])

x_test = x_train[-test_samples:]
x_train = x_train[:-test_samples]

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

plt.figure(figsize=(20,10))
plt.plot(np.array(df)[:,1])






# 1D Convolutional Autoencoder
print(len(x_train))
print(x_test.shape[1:])
print(np.prod(x_train.shape[1:]))
x_train_deep = x_train.reshape((len(x_train), int(np.prod(x_train.shape[1:]))))
x_test_deep = x_test.reshape((len(x_test), int(np.prod(x_test.shape[1:]))))
input_window = Input(shape=(window_length,1))
x = Conv1D(16, 3, activation="relu", padding="same")(input_window) # 10 dims
#x = BatchNormalization()(x)
x = MaxPooling1D(2, padding="same")(x) # 5 dims
x = Conv1D(1, 3, activation="relu", padding="same")(x) # 5 dims
#x = BatchNormalization()(x)
encoded = MaxPooling1D(2, padding="same")(x) # 3 dims

encoder = Model(input_window, encoded)

# 3 dimensions in the encoded layer

x = Conv1D(1, 3, activation="relu", padding="same")(encoded) # 3 dims
#x = BatchNormalization()(x)
x = UpSampling1D(2)(x) # 6 dims
x = Conv1D(16, 2, activation='relu')(x) # 5 dims
#x = BatchNormalization()(x)
x = UpSampling1D(2)(x) # 10 dims
decoded = Conv1D(1, 3, activation='sigmoid', padding='same')(x) # 10 dims
autoencoder = Model(input_window, decoded)
autoencoder.summary()

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
history = autoencoder.fit(x_train, x_train,
                epochs=epochs,
                batch_size=1024,
                shuffle=True,
                validation_data=(x_test, x_test))

decoded_stocks = autoencoder.predict(x_test)


plot_history(history)
plot_examples(x_test_deep, decoded_stocks)
plt.show()