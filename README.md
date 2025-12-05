# Midterm – Deep Learning  
Hands-On End-to-End Models (Classification, Regression & Clustering)

## 1. Repository Purpose
This repository contains my individual submission for the Deep Learning Midterm Examination (UTS) with the theme:

“Hands-On End-to-End Models in Machine Learning and Deep Learning”

The focus in this repository is on implementing **deep learning architectures** to solve three tasks:

1. Fraud detection (binary classification)
2. Song year prediction (regression)
3. Customer segmentation (clustering / representation learning)

Each task is built as a complete end-to-end Deep Learning pipeline, including data preprocessing, model construction, training, evaluation, and interpretation.

---

## 2. Project Overview

### 2.1 Objectives
The objectives of this project are as follows:

- Develop full Deep Learning pipelines for:
  - Binary classification (fraud detection)
  - Regression (song year prediction)
  - Unsupervised clustering (via learned representations or autoencoders)

- Apply key DL concepts:
  - Data preprocessing and normalization
  - Building neural network architectures (MLP, DNN, Autoencoder)
  - Model training using optimization algorithms (Adam, SGD, etc.)
  - Regularization techniques (dropout, batch normalization)
  - Basic hyperparameter tuning (learning rate, hidden layers, batch size)
  - Evaluating model performance and interpreting results

### 2.2 Task Implementations

#### (1) Fraud Detection – Binary Classification
- Goal: Predict whether a transaction is fraudulent (`isFraud = 1`) using a neural network classifier.
- Dataset includes various numerical and categorical transaction attributes.
- Outputs:
  - Neural network classification metrics (accuracy, F1, AUC)
  - Fraud probability predictions for the test set

#### (2) Song Year Prediction – Regression
- Goal: Predict the release year of a song using a deep regression model.
- Architecture typically includes:
  - Dense layers
  - Activation functions (ReLU, LeakyReLU)
  - Regularization layers if needed
- Outputs:
  - Regression metrics such as MAE, RMSE, and R²

#### (3) Customer Clustering – Deep Clustering / Autoencoder Approach
- Goal: Group customers based on spending and payment behavior.
- Method:
  - Train an autoencoder to learn compressed representations
  - Apply clustering (e.g., KMeans) on the latent space
- Outputs:
  - Cluster assignments
  - Interpretation of customer groups

---

## 3. Datasets

Same datasets as the Machine Learning midterm but implemented using Deep Learning models.

### 3.1 Fraud Detection Dataset
- `train_transaction.csv`  
  Contains labeled transaction records with `isFraud` as the target column.

- `test_transaction.csv`  
  Contains unlabeled transactions used for generating fraud probability predictions.

### 3.2 Regression Dataset
- `midterm-regresi-dataset.csv`  
  - First column = release year (target)  
  - Remaining columns = numeric audio features  

### 3.3 Clustering Dataset
- `clusteringmidterm.csv`  
  Contains customer-level features including balance, purchase patterns, cash advance usage, credit limit, payments, and tenure.

---

## 4. Project Structure
(Structure may be adjusted depending on the implementation.)

```text
midterm-deep-learning/
├── data/
│   ├── train_transaction.csv
│   ├── test_transaction.csv
│   ├── midterm-regresi-dataset.csv
│   └── clusteringmidterm.csv
├── notebooks/
│   ├── 01_fraud_detection_classification_dl.ipynb
│   ├── 02_song_year_regression_dl.ipynb
│   └── 03_customer_clustering_dl.ipynb
├── src/
│   ├── preprocessing.py
│   ├── dl_model_classification.py
│   ├── dl_model_regression.py
│   └── autoencoder_clustering.py
├── requirements.txt
└── README.md
