# Time-Series-LSTM-Forecasting

# Κ23γ: Ανάπτυξη Λογισμικού για Αλγοριθμικά Προβλήματα
## Χειμερινό εξάμηνο 2021-22 3η Προγραμματιστική Εργασία

### Ιωάννης Γεωργόπουλος, 1115201800026
### Βασίλειος Μαρκόπουλος, 1115201800108



# Περιγραφή Λειτουργίας

### Α) Πρόγνωση (forecasting) μελλοντικών τιμών μετοχής

![Screenshot from 2022-01-21 16-46-01](https://user-images.githubusercontent.com/83133031/150548174-cbf8dd87-2387-4625-8383-dc46a957ad21.png)

### Β) Ανίχνευση (detecting) σημαντικών αλλαγών στην τιμή μετοχής

![Screenshot from 2022-01-21 15-41-28](https://user-images.githubusercontent.com/83133031/150554711-b9413704-c6d3-4f19-bedc-4e63c2465559.png)

### Γ) Μείωση (reduction) της πολυπλοκότητας της μετοχής

![Screenshot from 2022-01-21 13-40-01](https://user-images.githubusercontent.com/83133031/150549409-eba91520-7f63-44a6-84df-8b936bb431d2.png)

# Δομή Εργασίας
Ο κώδικας που παραδίδεται αποτελείται από:
- Τα 3 κύρια προγράμματα:
  - forecast.py
  - detect.py
  - reduce.py
- 3 training προγράμματα που χρησιμοποιήθηκαν για εκπαίδευση μοντέλων.
- 3 προεκπαιδευμένα μοντέλα, ένα για κάθε πρόγραμμα.

Μαζί παραδίδονται ενδεικτικά input files από την εκφώνηση, και output files
του προγράμματος reduce.py, για να χρησιμοποιηθούν από τον κώδικα της
2ης εργασίας.

# Εκτέλεση Προγραμμάτων

### Α)
`python3 forecast.py -d <dataset> -n <number of time series selected> [-m <model directory>] [-noself] [-nomodel]`

Το πρόγραμμα διαλέγει τυχαία n χρονοσειρές από το σύνολο.

Τα -noself, -nomodel εκτελούν το πρόγραμμα χωρίς την εκτέλεση της πρόβλεψης με εκπαίδευση ανά χρονοσειρά, ή της πρόβλεψης από το σύνολο, αντίστοιχα.

By default, θα εκτελεστούν και οι δύο προβλέψεις.

Το όρισμα εκτέλεσης -m ακολουθείται από τη διαδρομή του αποθηκευμένου μοντέλου που θα χρησιμοποιηθεί.

Γρήγορη εκτέλεση: `python3 forecast.py -d input/nasdaq2007_17.csv -n 1 -m models/model_forecast`


### Β)
`python3 detect.py -d <dataset> -n <number of time series selected> -mae <error value as double> -m <model directory>`

Το πρόγραμμα διαλέγει τυχαία n χρονοσειρές από το σύνολο.

Το όρισμα εκτέλεσης -m ακολουθείται από τη διαδρομή του αποθηκευμένου μοντέλου που θα χρησιμοποιηθεί.

Γρήγορη εκτέλεση: `python3 detect.py -d input/nasdaq2007_17.csv -n 1 -mae 0.1 -m models/model_detect`


### Γ)
`python3 reduce.py -d <dataset> -q <queryset> -od <output_dataset_file> -oq <output_query_file> -m <model directory> [-novis]`

Το πρόγραμμα παράγει 2 αρχεία εξόδου, με όλα τα δεδομένα των αρχικών αρχείων σε reduced μορφή.

Το -novis όρισμα δεν παράγει γραφικές παραστάσεις για κάθε reduction, χρήσιμο εάν θέλουμε να παράγουμε γρήγορα output files,
αλλά όχι εάν θέλουμε να εξετάσουμε το σχήμα τους.

Το όρισμα εκτέλεσης -m ακολουθείται από τη διαδρομή του αποθηκευμένου μοντέλου που θα χρησιμοποιηθεί.

Γρήγορη εκτέλεση: `python3 reduce.py -d input/nasdaq2007_17_dataset.csv -q input/nasdaq2007_17_queryset.csv -od out/reduced_dataset -oq out/reduced_queryset -m models/model_reduce_enc`



Το πρόγραμμα διαλέγει τυχαία n χρονοσειρές από το σύνολο.
Εάν δεν δοθούν οι προαιρετικές μεταβλητές, εκπαιδεύει ένα νέο μοντέλο ανά χρονοσειρά,
και φορτώνει ένα προαποθηκευμένο μοντέλο 


# Αναφορά Πειραμάτων

### Α)
Αρχικά, ξεκινήσαμε εκπαιδεύοντας ένα μοντέλο με τις default τιμές.
Παρατηρήσαμε ότι, ενώ η εκπαίδευση ανά χρονοσειρά πήγαινε καλά,
το μοντέλο που εκπαιδευόταν ανά σύνολο χρονοσειρών δεν ήταν καλό,
πιθανώς λόγω overfitting. 
- lookback:   60
- batch_size: 32
- epochs:     100
- units:      50

![Screenshot from 2022-01-21 11-25-43](https://user-images.githubusercontent.com/83133031/150545515-2c1f22cb-eb4a-413b-889c-09327a993daa.png)

Επειδη ηταν αργο, αφαιρέσαμε 2 layers και μειώσαμε τις epochs στη μέση:
- lookback:   60
- batch_size: 32
- epochs:     50
- units:      50

![Screenshot from 2022-01-21 16-50-25](https://user-images.githubusercontent.com/83133031/150547553-d77c5b7d-4c1d-4bc6-81ea-9d451aadf42d.png)


Μετά την αποτυχία καταλάβαμε ότι τελικά χρειάζονται τα εξτρα layers, και για να φτιαξουμε
τον χρονο μειωσαμε ελαχιστα τα epochs και αυξησαμε το batch_size
- lookback:   60
- batch_size: 70
- epochs:     45
- units:      50

![Screenshot from 2022-01-21 12-27-05](https://user-images.githubusercontent.com/83133031/150545000-44e59272-fd00-466d-a0f0-b84b6fc0caae.png)
![Screenshot from 2022-01-21 16-46-01](https://user-images.githubusercontent.com/83133031/150548174-cbf8dd87-2387-4625-8383-dc46a957ad21.png)

Τελικα, φαινεται οτι τα αποτελεσματα ειναι ικανοποιητικα για τις περισσοτερες
χρονοσειρες που δοκιμασαμε, οποτε καταληξαμε σε αυτο το μοντελο.

### Β)
Ξεκινήσαμε με τις default τιμες:
- TIME_STEPS = 30
- unit_num = 64
- batch_size_num = 32
- epochs_num = 40
- MAE (THRESHOLD) = 1

![Screenshot from 2022-01-21 17-38-40](https://user-images.githubusercontent.com/83133031/150555607-09fdcb5e-4010-4df1-ba6e-5ee8ede756db.png)

Τελικά διαπιστώσαμε ότι το MAE πρέπει να είναι πολύ μικρότερο για να εντοπίζονται τα anomalies.
- TIME_STEPS = 30
- unit_num = 64
- batch_size_num = 64
- epochs_num = 40
- MAE (THRESHOLD) = 0.1


![Screenshot from 2022-01-21 15-41-28](https://user-images.githubusercontent.com/83133031/150554711-b9413704-c6d3-4f19-bedc-4e63c2465559.png)

### Γ)

Στο Γ, κρίνουμε την μετοχή από τις τιμές που παίρνει και από το σχήμα της.
Όταν φαίνεται από το γράφημα που παράγεται ότι η μετοχή είναι μια "συμπιεσμένη"
μορφή της αρχικής μετοχής, τότε θεωρούμε την μείωση επιτυχημένη, εφ' όσον οι τιμές
της νέας μετοχής είναι φυσιολογικές και όχι υπερβολικά χαμηλές ή υψηλές.

Ξεκινήσαμε χρησιμοποιώντας τις default τιμες
-window_length = 10
-encoding_dim = 3
-epochs = 100
-test_samples = 2000

Τα αποτελέσματα δεν ήταν καλά... ενώ η μετοχή που παραγόταν θύμιζε τη μετοχή μας σε σχήμα,
οι reduced τιμές ήταν πολύ μεγαλύτερες από τις αρχικές, σχεδόν 10πλάσιες!

![Screenshot from 2022-01-21 14-35-53](https://user-images.githubusercontent.com/83133031/150549093-ef5ecb24-affe-42a8-b1c4-ced8e9f35f0d.png)

![Screenshot from 2022-01-21 14-36-04](https://user-images.githubusercontent.com/83133031/150549591-c93d5548-2531-472b-8882-8cf632afa574.png)

Ύστερα δοκιμάσαμε τα:
-window_length = 10
-encoding_dim = 3
-epochs = 100
-test_samples = 300

![Screenshot from 2022-01-21 13-38-06](https://user-images.githubusercontent.com/83133031/150549185-1a7d4d2a-37a6-4bdb-a2a4-76598aeca75b.png)

![Screenshot from 2022-01-21 13-40-01](https://user-images.githubusercontent.com/83133031/150549409-eba91520-7f63-44a6-84df-8b936bb431d2.png)

Αυτές οι τιμές ήταν ιδανικές, αφού έχουμε αποτελέσματα που θυμίζουν τις αρχικές μετοχές, απλά σε reduced μορφή,
ενώ οι τιμές δεν έχουν μεγάλη απόκλιση από τις πραγματικές.

![Screenshot from 2022-01-21 15-03-57](https://user-images.githubusercontent.com/83133031/150549785-8b8ecfa6-277f-496f-ac0e-effe3dce78a9.png)

Εδώ εμφανίζονται δίπλα-δίπλα τα 2 αρχεία, για να φανεί ότι οι τιμές κάθε χρονοσειράς είναι παρόμοιες, δηλαδή ότι δεν αλλοιώνεται σοβαρά η αρχική.

### Δ)
Συγκρίνουμε τα αποτελέσματα και παρατηρούμε ότι:
To clustering δουλεύει παρόμοια και στις δύο περιπτώσεις, και παράγει παρόμοια
silhouette, επομένως δεν επηρεάζονται οι αριθμοί τόσο ώστε να χαλάσει το clustering.
Από την άλλη, το search παρουσιάζει διαφορετικά αποτελέσματα. Συγκεκριμένα
τα reduced δεδομένα έχουν λίγο χειρότερο (μεγαλύτερο) MAF, όμως ο χρόνος μειώνεται
δραματικά, ιδίως για τις Frechet μεθόδους, που καθυστερούσαν υπερβολικά πολύ
με τα δεδομένα μεγέθους 3650.

Εν ολίγοις, θα προτείναμε τη χρήση των reduced δεδομένων, αφού ο χρόνος που εξοικονομούν
είναι σημαντικότερος από την μικρή αλλοίωση των αποτελεσμάτων.

Αποσπάσματα από τα output files της 2ης Εργασίας με τα κανονικά αρχεία
nasdaq2007_17.csv και με τα reduced datasets που παρήχθησαν στο ερώτημα Γ:

![Screenshot from 2022-01-21 11-47-03](https://user-images.githubusercontent.com/83133031/150551243-16ab7b3a-4c11-49ce-a332-87aea27418e0.png)
![Screenshot from 2022-01-21 11-56-13](https://user-images.githubusercontent.com/83133031/150551288-a3acc23d-3e9e-4a17-bff2-511d81315cc9.png)
![Screenshot from 2022-01-21 11-57-28](https://user-images.githubusercontent.com/83133031/150551317-92ae2f1a-0423-456f-9fc8-728c3b77bc46.png)


![Screenshot from 2022-01-21 11-43-40](https://user-images.githubusercontent.com/83133031/150551546-7f98a492-d163-46a9-af1d-3529e06af1d4.png)
![Screenshot from 2022-01-21 11-44-04](https://user-images.githubusercontent.com/83133031/150551549-42208630-7ee9-4b7d-aa29-46225b7cb3f5.png)
![Screenshot from 2022-01-21 11-44-17](https://user-images.githubusercontent.com/83133031/150551551-f06771af-1472-48b1-b03b-f83afedd560a.png)
![Screenshot from 2022-01-21 11-44-53](https://user-images.githubusercontent.com/83133031/150551554-4f071967-1578-4c2a-af5b-bb2d18c7094e.png)
![Screenshot from 2022-01-21 11-45-03](https://user-images.githubusercontent.com/83133031/150551559-fb24f701-5cea-4e08-b8be-c58e68c7651e.png)
![Screenshot from 2022-01-21 11-45-14](https://user-images.githubusercontent.com/83133031/150551561-b3671610-dfa1-42a2-a809-7787da08057a.png)
![Screenshot from 2022-01-21 11-45-29](https://user-images.githubusercontent.com/83133031/150551564-34621d29-34cf-466d-a16f-6902f3b6e4eb.png)


Για την εκτέλεση των αρχείων της Εργασίας 2 με τις τιμές μας, χρησιμοποιούνται οι παρακάτω εντολές:
Δ)
Search:

`./bin/search.out -i input/nasdaq2007_17_dataset.csv -q input/nasdaq2007_17_queryset.csv -o outputFiles/search_lsh.txt -algorithm LSH`

`./bin/search.out -i input/nasdaq2007_17_dataset.csv -q input/nasdaq2007_17_queryset.csv -o outputFiles/search_hypercube.txt -algorithm Hypercube`

`./bin/search.out -i input/nasdaq2007_17_dataset.csv -q input/nasdaq2007_17_queryset_one.csv -o outputFiles/search_fr_d.txt -algorithm Frechet -metric discrete`

`./bin/search.out -i input/nasdaq2007_17_dataset.csv -q input/nasdaq2007_17_queryset_one.csv -o outputFiles/search_fr_c.txt -algorithm Frechet -metric continuous`


`./bin/search.out -i input/dataset.csv -q input/queryset.csv -o outputFiles/reduced_search_lsh.txt -algorithm LSH`

`./bin/search.out -i input/dataset.csv -q input/queryset.csv -o outputFiles/recuded_search_hypercube.txt -algorithm Hypercube`

`./bin/search.out -i input/dataset.csv -q input/queryset_one.csv -o outputFiles/reduced_search_fr_d.txt -algorithm Frechet -metric discrete`

`./bin/search.out -i input/dataset.csv -q input/queryset_one.csv -o outputFiles/reduced_search_fr_c.txt -algorithm Frechet -metric continuous`

Clustering:
Classic - Vector

`./bin/cluster.out -i input/nasdaq2007_17_dataset.csv -c input/cluster.conf -o outputFiles/cluster_classic_vector.txt -assignment Classic -update Mean_Vector -silhouette`

Classic - Frechet

`./bin/cluster.out -i input/nasdaq2007_17_dataset.csv -c input/cluster.conf -o outputFiles/cluster_classic_frechet.txt -assignment Classic -update Mean_Frechet -silhouette`

LSH Frechet - Mean Frechet

`./bin/cluster.out -i input/nasdaq2007_17_dataset.csv -c input/cluster.conf -o outputFiles/cluster_lsh_frechet.txt -assignment LSH_Frechet -update Mean_Frechet -silhouette`

LSH Vector - Vector

`./bin/cluster.out -i input/nasdaq2007_17_dataset.csv -c input/cluster.conf -o outputFiles/cluster_lsh_vector.txt -assignment LSH -update Mean_Vector -silhouette`

Hypercube - Vector

`./bin/cluster.out -i input/nasdaq2007_17_dataset.csv -c input/cluster.conf -o outputFiles/cluster_hypercube.txt -assignment Hypercube -update Mean_Vector -silhouette`

Our Values:
Clustering:
Classic - Vector

`./bin/cluster.out -i input/dataset.csv -c input/cluster.conf -o outputFiles/reduced_cluster_classic_vector.txt -assignment Classic -update Mean_Vector -silhouette`

Classic - Frechet

`./bin/cluster.out -i input/dataset.csv -c input/cluster.conf -o outputFiles/reduced_cluster_classic_frechet.txt -assignment Classic -update Mean_Frechet -silhouette`

LSH Frechet - Mean Frechet

`./bin/cluster.out -i input/dataset.csv -c input/cluster.conf -o outputFiles/reduced_cluster_lsh_frechet.txt -assignment LSH_Frechet -update Mean_Frechet -silhouette`

LSH Vector - Vector

`./bin/cluster.out -i input/dataset.csv -c input/cluster.conf -o outputFiles/reduced_cluster_lsh_vector.txt -assignment LSH -update Mean_Vector -silhouette`

Hypercube - Vector

`./bin/cluster.out -i input/dataset.csv -c input/cluster.conf -o outputFiles/reduced_cluster_hypercube.txt -assignment Hypercube -update Mean_Vector -silhouette`


Πηγές: 
- Forecasting: https://towardsdatascience.com/lstm-time-series-forecasting-predicting-stock-prices-using-an-lstm-model-6223e9644a2f 
- Anomaly Detection: https://curiousily.com/posts/anomaly-detection-in-time-series-with-lstms-using-keras-in-python/  
- Dimensionality Reduction: https://towardsdatascience.com/autoencoders-for-the-compression-of-stock-market-data-28e8c1a2da3e
