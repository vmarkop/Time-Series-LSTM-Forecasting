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
    print("Usage: python reduce.py -d <dataset> -q <queryset> -od <output_dataset_file> -oq <output_query_file> -m <model directory>")
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

        scaler = MinMaxScaler()
        x_before_train = scaler.fit_transform(df['price'].values.reshape(-1,1))

        x_timeseries = np.array([x_before_train[i-window_length:i] for i in tqdm(range(window_length+1,len(df['price']),window_length))])
        
        # 1D Convolutional Autoencoder
        encoder = load_model("model_enc")

        encoded_stock = encoder.predict(x_timeseries)
        encoded_stock = encoded_stock.reshape(1,-1)
        encoded_stock = scaler.inverse_transform(encoded_stock)

        output = open(output_file, "a")
        output.write(stock)
        for i in encoded_stock[0]:
            output.write('\t')
            output.write(str(i))
        output.write('\n')
        output.close()

        # Visualising the results
        # plt.plot(range(3650),df['price'].values, color = 'red', label = 'Real TESLA Stock Price')
        # plt.plot(range(1092),encoded_stock.reshape(-1,1), color = 'blue', label = 'Reduced TESLA Stock Price')
        # plt.xticks(np.arange(0,3650,365))
        # plt.title('TESLA Stock Price Prediction')
        # plt.xlabel('Time')
        # plt.ylabel('TESLA Stock Price')
        # plt.legend()
        # plt.show()

# Produce output_dataset_file
reduce(dataset,data_out)

# Produce output_queryset_file
reduce(queryset,query_out)