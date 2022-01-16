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





# Parsing input parameters
if len(sys.argv) < 9:
    print("Usage: python reduce.py -d <dataset> -q <queryset> -od <output_dataset_file> -oq <output_query_file>")
    sys.exit()

wrong_arg = False
if sys.argv[1] == "-d":
    dataset = sys.argv[2]
else:
    wrong_arg = True
if sys.argv[3] == "-q":
    queryset = sys.argv[4]
else:
    wrong_arg = True
if sys.argv[5] == "-od":
    data_out = sys.argv[6]
    open(data_out, 'w').close()
else:
    wrong_arg = True
if sys.argv[7] == "-oq":
    query_out = sys.argv[8]
    open(query_out, 'w').close()
else:
    wrong_arg = True

if wrong_arg:
    print("Usage: python reduce.py -d <dataset> -q <queryset> -od <output_dataset_file> -oq <output_query_file>")
    sys.exit()


def reduce(input_file, output_file):

    dataframe = pd.read_csv(input_file,'\t',header=None).iloc[:,:]

    autoencoder = Model()


    for ts in range(len(dataframe)):

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

        if ts == 0:
            encoder = Model(input_window, encoded)

        # 3 dimensions in the encoded layer

        x = Conv1D(1, 3, activation="relu", padding="same")(encoded) # 3 dims
        #x = BatchNormalization()(x)
        x = UpSampling1D(2)(x) # 6 dims
        x = Conv1D(16, 2, activation='relu')(x) # 5 dims
        #x = BatchNormalization()(x)
        x = UpSampling1D(2)(x) # 10 dims
        decoded = Conv1D(1, 3, activation='sigmoid', padding='same')(x) # 10 dims
        
        if ts == 0:
            autoencoder = Model(input_window, decoded)
        autoencoder.summary()

        autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
        history = autoencoder.fit(x_train, x_train,
                        epochs=epochs,
                        batch_size=1024,
                        shuffle=True,
                        validation_data=(x_test, x_test))

        decoded_stocks = autoencoder.predict(x_test)


        # output = open(output_file, "a")
        # output.write(stock)
        # for i in decoded_stocks[1]:
        #     output.write('\t')
        #     output.write(str(i[0]))
        # output.write('\n')
        # output.close()

    # plot_history(history)
    # plot_examples(x_test_deep, decoded_stocks)
    # plt.show()
    autoencoder.save('model_reduce')

# Produce output_dataset_file
reduce(dataset,data_out)

# Produce output_queryset_file
# reduce(queryset,query_out)