MEX Classification Project
Author: Ana Arsić
Dataset: MEx (Multimodal Exercise Dataset)
Source: https://archive.ics.uci.edu/dataset/500/mex

--------------------------------------------
OPIS PROJEKTA
--------------------------------------------
U ovom projektu rešavan je višeklasni problem klasifikacije
tipa fizičke vežbe korišćenjem MEx skupa podataka.

Korišćeni su različiti skupovi atributa:
- Full (svi atributi)
- PCA90 (95% očuvane varijanse)
- PCA95 (95% očuvane varijanse)
- KBest50 (selekcija atributa pomoću mutual information)

Primena algoritama:
- Decision Tree
- Random Forest
- Logistic Regression
- SVM
- KNN
- Naive Bayes

Evaluacija modela izvršena je korišćenjem:
- Accuracy
- F1-macro
- Confusion matrix (train i test skup)

--------------------------------------------
SADRŽAJ FOLDERA
--------------------------------------------

DATA FAJLOVI:
- mex_features_all_raw.csv
  Skup podataka nakon feature engineering-a (pre preprocesiranja).

- mex_features_all_preprocessed.csv
  Skup podataka nakon čišćenja i imputacije nedostajućih vrednosti.

PODACI O PODELI:
- train_test_split_indices.npz
  Indeksi train i test skupa (radi potpune reproduktivnosti).

- class_distribution_train.csv
  Raspodela klasa u train skupu.

- class_distribution_test.csv
  Raspodela klasa u test skupu.

REZULTATI MODELA:
- model_results_all.csv
  Rezultati svih modela za sve varijante atributa.

- best_per_variant.csv
  Najbolji model za svaku varijantu atributa.

SAČUVANI MODELI:
- best_model_overall__*.joblib
  Najbolji model ukupno (pipeline: preprocessing + redukcija + klasifikator).

- best_model__*.joblib
  Najbolji model po varijanti atributa.

--------------------------------------------
REPRODUKCIJA
--------------------------------------------
Svi modeli su sačuvani kao sklearn pipeline objekti.
Mogu se učitati pomoću:

    import joblib
    model = joblib.load("best_model_overall__...joblib")
    predictions = model.predict(X)

Za ponavljanje eksperimenta potrebno je pokrenuti
class_mex.ipynb u lokalnom Python okruženju sa instaliranim
sledećim bibliotekama:

- numpy
- pandas
- scikit-learn
- matplotlib
- joblib

--------------------------------------------
NAPOMENA
--------------------------------------------
Korišćena je stratifikovana podela skupa podataka
kako bi se očuvala raspodela klasa između train i test skupa.
