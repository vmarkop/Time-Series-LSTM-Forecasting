import sys
import math
import random
import matplotlib.pyplot as plt
import keras
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping

import os
# os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2/bin")

# Parsing input parameters
if len(sys.argv) >= 5 and sys.argv[1] == "-d" and sys.argv[3] == "-o":
    dataset = sys.argv[2]
    output = sys.argv[4]
else:
    print("Usage: python forecast_trainall.py -d <dataset> -o <output>")
    sys.exit()

dataframe = pd.read_csv(dataset,'\t',header=None).iloc[:,:]
model = Sequential()

for ts in range(0, len(dataframe)):
    stock = dataframe.iloc[ts][0]
    df = dataframe.iloc[ts,1:]
    training_num = math.ceil(df.size*0.8)
    testing_num = df.size - training_num

    # Training variables
    lookback = 60
    batch_size_num = 64
    epoch_num = 40
    unit_num = 50

    training_set = df.iloc[:training_num].values
    test_set     = df.iloc[training_num:].values

    # Feature Scaling
    sc = MinMaxScaler(feature_range = (0, 1))
    training_set_scaled = sc.fit_transform(training_set.reshape(-1,1))

    # Creating a data structure with 60 time-steps and 1 output
    X_train = []
    y_train = []
    for i in range(lookback, training_num):
        X_train.append(training_set_scaled[i-lookback:i, 0])
        y_train.append(training_set_scaled[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    #(training_num-lookback, lookback, 1)

    # Only execute on first loop iteration
    if ts == 0:

        #Adding the first LSTM layer and some Dropout regularisation
        model.add(LSTM(units = unit_num, return_sequences = True, input_shape = (X_train.shape[1], 1)))
        model.add(Dropout(0.2))

        # Adding a second LSTM layer and some Dropout regularisation
        model.add(LSTM(units = unit_num, return_sequences = True))
        model.add(Dropout(0.2))

        # Adding a third LSTM layer and some Dropout regularisation
        model.add(LSTM(units = unit_num, return_sequences = True))
        model.add(Dropout(0.2))

        # Adding a fourth LSTM layer and some Dropout regularisation
        model.add(LSTM(units = unit_num))
        model.add(Dropout(0.2))

        # Adding the output layer
        model.add(Dense(units = 1))

        # Compiling the RNN
        model.compile(optimizer = 'adam', loss = 'mean_squared_error')

    # Fitting the RNN to the Training set
    model.fit(X_train, y_train, epochs = epoch_num, batch_size = batch_size_num)

    # Backup every 50 timeseries
    if ts % 50 == 0:
        model.save(output)

model.save(output)