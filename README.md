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
Search:
./bin/search.out -i input/nasdaq2007_17_dataset.csv -q input/nasdaq2007_17_queryset.csv -o outputFiles/search_lsh.txt -algorithm LSH
./bin/search.out -i input/nasdaq2007_17_dataset.csv -q input/nasdaq2007_17_queryset.csv -o outputFiles/search_hypercube.txt -algorithm Hypercube
./bin/search.out -i input/nasdaq2007_17_dataset.csv -q input/nasdaq2007_17_queryset_one.csv -o outputFiles/search_fr_d.txt -algorithm Frechet -metric discrete
./bin/search.out -i input/nasdaq2007_17_dataset.csv -q input/nasdaq2007_17_queryset_one.csv -o outputFiles/search_fr_c.txt -algorithm Frechet -metric continuous

./bin/search.out -i input/dataset.csv -q input/queryset.csv -o outputFiles/reduced_search_lsh.txt -algorithm LSH
./bin/search.out -i input/dataset.csv -q input/queryset.csv -o outputFiles/recuded_search_hypercube.txt -algorithm Hypercube
./bin/search.out -i input/dataset.csv -q input/queryset_one.csv -o outputFiles/reduced_search_fr_d.txt -algorithm Frechet -metric discrete
./bin/search.out -i input/dataset.csv -q input/queryset_one.csv -o outputFiles/reduced_search_fr_c.txt -algorithm Frechet -metric continuous

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

Our Values:
Clustering:
Classic - Vector
./bin/cluster.out -i input/dataset.csv -c input/cluster.conf -o outputFiles/reduced_cluster_classic_vector.txt -assignment Classic -update Mean_Vector -silhouette

Classic - Frechet
./bin/cluster.out -i input/dataset.csv -c input/cluster.conf -o outputFiles/reduced_cluster_classic_frechet.txt -assignment Classic -update Mean_Frechet -silhouette

LSH Frechet - Mean Frechet
./bin/cluster.out -i input/dataset.csv -c input/cluster.conf -o outputFiles/reduced_cluster_lsh_frechet.txt -assignment LSH_Frechet -update Mean_Frechet -silhouette

LSH Vector - Vector
./bin/cluster.out -i input/dataset.csv -c input/cluster.conf -o outputFiles/reduced_cluster_lsh_vector.txt -assignment LSH -update Mean_Vector -silhouette

Hypercube - Vector
./bin/cluster.out -i input/dataset.csv -c input/cluster.conf -o outputFiles/reduced_cluster_hypercube.txt -assignment Hypercube -update Mean_Vector -silhouette


Α) 
`python forecast.py -d <dataset> -n <number of time series selected> [-m <model directory>] [-noself] [-nomodel]`

Το όρισμα εκτέλεσης -m ακολουθείται από τη διαδρομή του αποθηκευμένου
μοντέλου που θα χρησιμοποιηθεί. Επίσης μπορούν προαιρετικά
να χρησιμοποιηθούν τα ορίσματα -noself για να μην γίνει εκπαίδευση
ανά χρονοσειρά, και -nomodel.

Το πρόγραμμα διαλέγει τυχαία n χρονοσειρές από το σύνολο.
Εάν δεν δοθούν οι προαιρετικές μεταβλητές, εκπαιδεύει ένα νέο μοντέλο ανά χρονοσειρά,
και φορτώνει ένα προαποθηκευμένο μοντέλο 

temp_model bad!
dont forget maybe to add layers back!

# Πειράματα

### Α)
Αρχικά, ξεκινήσαμε εκπαιδεύοντας ένα μοντέλο με τις default τιμές.
Παρατηρήσαμε ότι, ενώ η εκπαίδευση ανά χρονοσειρά πήγαινε καλά,
το μοντέλο που εκπαιδευόταν ανά σύνολο χρονοσειρών δεν ήταν καλό,
πιθανώς λόγω overfitting. 
(τιμη1) 60 32 100 50
(τιμη2)
(τιμη3)
(πλοτ που εχει κανονικη, πρεδικτεδ και πρεδικτεδ_αλλ)

Επειδη ηταν αργο, αλλαξαμε αυτες τις τιμες
(τιμη1) αφαιρέσαμε 2 layers, 50 epochs
(πλοτ).

Σκεφτηκαμε ότι τελικά χρειαζόμαστε τα εξτρα λαυερς, και για να φτιαξουμε
τον χρονο μειωσαμε ελαχιστα τα εποχς και αυξησαμε τα μπατςις
(τιμη)
(πλοτ)
45 εποχ
70 μπατς

Τελικα, φαινεται οτι τα αποτελεσματα ειναι ικανοποιητικα για τις περισσοτερες
χρονοσειρες που δοκιμασαμε, οποτε καταληξαμε σε αυτο το μοντελο.
lookback:   60
batch_size: 70
epochs:     45
units:      50

### Β)
Κι εδώ ξεκινήσαμε με τις default τιμες
(τιμη)
(πλοτ)
TIME_STEPS = 30
unit_num = 64
batch_size_num = 32
epochs_num = 40

Telika 
TIME_STEPS = 30
unit_num = 64
batch_size_num = 64
epochs_num = 40


### Γ)
Ξεκινησαμε με default τιμες
2000 test samples και 100 epochs.
Το window length και το encoding dimension φάνηκαν ιδανικά εξ αρχής
οπότε δεν τα πειράξαμε.
(τιμη)
(πλοτ)

Δεν δούλευε καθόλου το reduction, όπως φαίνεται,
δηλαδή η μετοχή έπαιρνε παντού όλες τις τιμές
και σχημάτιζε μια ευθεία γραμμή.
μετα απο τεστινγ καταληξαμε σε αυτα
window_length = 10
encoding_dim = 3
epochs = 100
test_samples = 300
(πλοτ)

model_enc = good

Κρίνουμε την μετοχή από τις τιμές που παίρνει και από το σχήμα της
Φαίνεται από το γράφημα που παράγεται ότι η μετοχή είναι μια "συμπιεσμένη"
μορφή της αρχικής μετοχής, οπότε θεωρούμε την μείωση επιτυχημένη.
Επίσης, σε αυτήν την φωτογραφία φαίνονται οι τιμές,
και παρατηρούμε ότι η νέα μετοχή παίνρει φυσιολογικές τιμές,
κοντά στις αρχικές.


### Δ)
Συγκρίνουμε τα αποτελέσματα και παρατηρούμε ότι:
To clustering δουλεύει παρόμοια και στις δύο περιπτώσεις, και παράγει παρόμοια
silhouette, επομένως δεν επηρεάζονται οι αριθμοί τόσο ώστε να χαλάσει το clustering.
Από την άλλη, το search παρουσιάζει διαφορετικά αποτελέσματα. Συγκεκριμένα
τα reduced δεδομένα έχουν λίγο χειρότερο (μεγαλύτερο) MAF, όμως ο χρόνος μειώνεται
δραματικά, ιδίως για τις Frechet μεθόδους, που καθυστερούσαν υπερβολικά πολύ
με τα δεδομένα μεγέθους 3650.

Αποσπάσματα από τα output files της 2ης Εργασίας με τα κανονικά αρχεία
nasdaq2007_17.csv και με τα reduced datasets που παρήχθησαν στο ερώτημα Γ: