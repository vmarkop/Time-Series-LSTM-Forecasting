# Time-Series-LSTM-Forecasting

# Κ23γ: Ανάπτυξη Λογισμικού για Αλγοριθμικά Προβλήματα
## Χειμερινό εξάμηνο 2021-22 3η Προγραμματιστική Εργασία

### Ιωάννης Γεωργόπουλος, 1115201800026
### Βασίλειος Μαρκόπουλος, 1115201800108

https://towardsdatascience.com/lstm-time-series-forecasting-predicting-stock-prices-using-an-lstm-model-6223e9644a2f


 
- Forecasting: https://towardsdatascience.com/lstm-time-series-forecasting-predicting-stock-prices-using-an-lstm-model-6223e9644a2f 
- Anomaly Detection: https://curiousily.com/posts/anomaly-detection-in-time-series-with-lstms-using-keras-in-python/  
- Dimensionality Reduction: https://towardsdatascience.com/autoencoders-for-the-compression-of-stock-market-data-28e8c1a2da3e


Δ)
Clustering:
Classic - Vector
./bin/cluster.out -i input/nasdaq2007_17_dataset.csv -c input/cluster.conf -o outputFiles/cluster_classic_vector.txt -assignment Classic -update Mean_Vector -silhouette

Classic - Frechet
./bin/cluster.out -i input/nasdaq2007_17_dataset.csv -c input/cluster.conf -o outputFiles/cluster_classic_frechet.txt -assignment Classic -update Mean_Frechet -silhouette

LSH Frechet - Mean Frechet
./bin/cluster.out -i input/nasdaq2007_17_dataset.csv -c input/cluster.conf -o outputFiles/cluster_lsh_frechet.txt -assignment LSH_Frechet -update Mean_Frechet -silhouette

LSH Vector - Vector
./bin/cluster.out -i input/nasdaq2007_17_dataset.csv -c input/cluster.conf -o outputFiles/cluster_lsh_vector.txt -assignment LSH -update Mean_Vector -silhouette

Hypercube - Vector
./bin/cluster.out -i input/nasdaq2007_17_dataset.csv -c input/cluster.conf -o outputFiles/cluster_hypercube.txt -assignment Hypercube -update Mean_Vector -silhouette
