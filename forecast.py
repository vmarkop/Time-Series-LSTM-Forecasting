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

# Parsing input parameters
if len(sys.argv) >= 3 and sys.argv[1] == "-d":
    dataset = sys.argv[2]
else:
    print("Usage: python forecast.py -d <dataset> -n <number of time series selected> [-m <model directory>] [-noself] [-nomodel]")
    sys.exit()
n = 1
if len(sys.argv) >= 5 and sys.argv[3] == "-n":
    n = int(sys.argv[4])
saved_model = "models/model_forecast"
if len(sys.argv) >= 7 and sys.argv[5] == "-m":
    saved_model = sys.argv[6]
elif len(sys.argv) >= 8 and sys.argv[6] == "-m":
    saved_model = sys.argv[7]
noself = False
nomodel = False
if "-noself" in sys.argv:
    noself = True
if "-nomodel" in sys.argv:
    nomodel = True

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

    if not noself:
        model = Sequential()

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

    # Getting the predicted stock price of 2017
    dataset_train = df.iloc[:training_num]
    dataset_test = df.iloc[training_num:]

    dataset_total = pd.concat((dataset_train, dataset_test), axis = 0)

    inputs = dataset_total[len(dataset_total) - len(dataset_test) - lookback:].values

    inputs = inputs.reshape(-1,1)
    inputs = sc.transform(inputs)
    X_test = []
    for i in range(lookback, testing_num+lookback):
        X_test.append(inputs[i-lookback:i, 0])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    if not noself:
        predicted_stock_price = model.predict(X_test)
        predicted_stock_price = sc.inverse_transform(predicted_stock_price)

    if not nomodel:
        model_all = keras.models.load_model(saved_model)
        predicted_all_stock_price = model_all.predict(X_test)
        predicted_all_stock_price = sc.inverse_transform(predicted_all_stock_price)

    # Visualising the results

    plt.plot(range(training_num,df.size),dataset_test.values, color = "red", label = "Real " + stock + " stock price")
    if not noself:
        plt.plot(range(training_num,df.size),predicted_stock_price, color = "blue", label = "Predicted " + stock + " stock price")
    if not nomodel:
        plt.plot(range(training_num,df.size),predicted_all_stock_price, color = "green", label = "Predicted " + stock + " stock price with model_all")
    plt.xticks(np.arange(training_num,df.size,100))
    plt.title(stock + ' Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel(stock + ' Stock Price')
    plt.legend()
    plt.show()