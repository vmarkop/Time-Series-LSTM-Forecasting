from keras.layers import Input, Conv1D, MaxPooling1D, UpSampling1D
from keras.models import Model, load_model
import sys
import requests as req
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import matplotlib.pyplot as plt

# Parameters
window_length = 10
encoding_dim = 3
epochs = 100
test_samples = 300

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
saved_model = "models/model_reduce"
if len(sys.argv) >= 11:
    if sys.argv[9] == "-m":
        saved_model = sys.argv[10]

if wrong_arg:
    print("Usage: python reduce.py -d <dataset> -q <queryset> -od <output_dataset_file> -oq <output_query_file> -m <model directory>")
    sys.exit()


def reduce(input_file, output_file):

    dataframe = pd.read_csv(input_file,'\t',header=None).iloc[:,:]

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

        encoder = Model(input_window, encoded)
        # 3 dimensions in the encoded layer

        x = Conv1D(1, 3, activation="relu", padding="same")(encoded) # 3 dims
        #x = BatchNormalization()(x)
        x = UpSampling1D(2)(x) # 6 dims
        x = Conv1D(16, 2, activation='relu')(x) # 5 dims
        #x = BatchNormalization()(x)
        x = UpSampling1D(2)(x) # 10 dims
        decoded = Conv1D(1, 3, activation='sigmoid', padding='same')(x) # 10 dims

        autoencoder = load_model(saved_model)
        decoded_stocks = autoencoder.predict(x_test)

        output = open(output_file, "a")
        output.write(stock)
        for i in decoded_stocks:
            for j in i:
                output.write('\t')
                output.write(str(j[0]))
        output.write('\n')
        output.close()

# Produce output_dataset_file
reduce(dataset,data_out)

# Produce output_queryset_file
reduce(queryset,query_out)