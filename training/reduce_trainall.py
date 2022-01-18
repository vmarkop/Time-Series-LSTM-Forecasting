from keras.layers import Input, Conv1D, MaxPooling1D, UpSampling1D
from keras.models import Model
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
if len(sys.argv) >= 5 and sys.argv[1] == "-d" and sys.argv[3] == "-o":
    dataset = sys.argv[2]
    output = sys.argv[4]
else:
    print("Usage: python reduce_trainall.py -d <dataset> -o <output>")
    sys.exit()


def reduce(input_file, output_file):

    dataframe = pd.read_csv(input_file,'\t',header=None).iloc[:,:]

    autoencoder = Model()


    for ts in range(0, len(dataframe)):

        timeseries = dataframe.iloc[ts,1:]
        timeseries = timeseries.values.reshape(-1,1)
        df = pd.DataFrame(timeseries, columns=['price'])
        df.price = df.price.astype('int')
        # df['pct_change'] = df.price.pct_change()
        # df['log_ret'] = np.log(df.price) - np.log(df.price.shift(1))

        scaler = MinMaxScaler()
        x_train = np.array([scaler.fit_transform(df['price'].values[i-window_length:i].reshape(-1, 1)) for i in tqdm(range(window_length+1,len(df['price'])))])

        x_test = x_train[-test_samples:]
        x_train = x_train[:-test_samples]

        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')


        # 1D Convolutional Autoencoder
        input_window = Input(shape=(window_length,1))
        x = Conv1D(16, 3, activation="relu", padding="same")(input_window) # 10 dims
        x = MaxPooling1D(2, padding="same")(x) # 5 dims
        x = Conv1D(1, 3, activation="relu", padding="same")(x) # 5 dims
        encoded = MaxPooling1D(2, padding="same")(x) # 3 dims

        # 3 dimensions in the encoded layer

        x = Conv1D(1, 3, activation="relu", padding="same")(encoded) # 3 dims
        x = UpSampling1D(2)(x) # 6 dims
        x = Conv1D(16, 2, activation='relu')(x) # 5 dims
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


        # Backup every 50 timeseries
        if ts % 50 == 0:
            autoencoder.save(output)

    autoencoder.save(output)


reduce(dataset,"None")