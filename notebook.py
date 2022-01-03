import sys
import math
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
# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = "0"

# Parsing input parameters
if len(sys.argv) >= 3 and sys.argv[1] == "-d":
    dataset = sys.argv[2]
else:
    print("Usage: python forecast.py -d <dataset> -n <number of time series selected> -all [optional]")
    sys.exit()
n = 1
if len(sys.argv) >= 5 and sys.argv[3] == "-n":
    n = sys.argv[4]
all = True if len(sys.argv) > 5 and sys.argv[5] == "-all" else False




df=pd.read_csv(dataset,'\t',header=None).iloc[:,1:]
training_num = math.ceil(len(df.columns)*0.8)
print("Number of rows and columns:", df.shape)
print(df.head(5))

df=df.T
df.insert(loc=0,column='dates',value=list(range(1,len(df.index)+1)))
print(df)
timeseries_num = 0

training_set = df.iloc[:training_num, 1:2].values
test_set = df.iloc[training_num:, 1:2].values

# Feature Scaling
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)# Creating a data structure with 60 time-steps and 1 output
X_train = []
y_train = []
for i in range(60, math.ceil(0.65*training_num)):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
#(740, 60, 1)

model = Sequential()#Adding the first LSTM layer and some Dropout regularisation
model.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
model.add(Dropout(0.2))# Adding a second LSTM layer and some Dropout regularisation
model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))# Adding a third LSTM layer and some Dropout regularisation
model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))# Adding a fourth LSTM layer and some Dropout regularisation
model.add(LSTM(units = 50))
model.add(Dropout(0.2))# Adding the output layer
model.add(Dense(units = 1))

# Compiling the RNN
model.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
model.fit(X_train, y_train, epochs = 1, batch_size = 32)

# Getting the predicted stock price of 2017
dataset_train = df.iloc[:training_num, 1:2]
dataset_test = df.iloc[training_num:, 1:2]

dataset_total = pd.concat((dataset_train, dataset_test), axis = 0)

inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range(60, 519):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

print(X_test.shape)
# (459, 60, 1)

predicted_stock_price = model.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

dates = df.loc[(training_num):,"dates"]#.values.reshape(-1,1)

# Visualising the results
plt.plot(dates,dataset_test.values[:,0], color = "red", label = "Real TESLA Stock Price")
plt.plot(dates,predicted_stock_price, color = "blue", label = "Predicted TESLA Stock Price")
plt.xticks(np.arange(0,459,50))
plt.title('TESLA Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('TESLA Stock Price')
plt.legend()
plt.show()