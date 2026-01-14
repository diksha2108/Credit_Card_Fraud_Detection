# Credit_Card_Fraud_Detection

Credit Card Fraud Detection: A Complete ML Pipeline
A Comparative Study of Resampling and Modeling Strategies

An end-to-end machine learning pipeline for detecting fraudulent credit card transactions, leveraging advanced resampling techniques and ensemble modeling to tackle class imbalance and improve predictive performance.

Overview
This project addresses credit card fraud detection using supervised and unsupervised machine learning on a highly imbalanced dataset (~0.18% fraud). It builds a complete ML pipeline covering preprocessing, exploratory analysis, feature engineering, and advanced resampling.

To tackle class imbalance and improve performance, we applied techniques like random undersampling, SMOTE, and a custom KMeans-based method, paired with models such as logistic regression, random forest, KNN, neural networks, and an ensemble voting classifier.

The best-performing model was selected through threshold tuning and F1-score optimization, followed by targeted feature engineering to further boost performance. The pipeline is modular, reproducible, and CLI-configurable—ready for real-world deployment.

Dataset
Source: Kaggle - Credit Card Fraud Detection

Total Transactions: 284,807

Fraudulent Cases: 492

Features:

V1 to V28 (PCA-transformed)
Amount, Time
Class (target: 1 = fraud, 0 = legitimate)
Key Highlights
Implemented six resampling techniques to address class imbalance:

baseline (cost-sensetive)
Random Undersampling
NearMiss
Custom KMeans-based undersampling
Random Oversampling
SMOTE
Trained and compared five machine learning models:

Logistic Regression
Random Forest
K-Nearest Neighbors
Neural Network (MLPClassifier)
Ensemble Voting Classifier combining top performers
Custom KMeans-based undersampler developed to improve representation of the majority class while preserving data structure

Feature engineering to enhance model performance, supported by feature importance analysis to identify influential variables

Applied threshold tuning based on F1-score to better balance precision and recall

Modular, CLI-driven architecture for reproducible and configurable experimentation

Professional final report and cleanly structured Jupyter notebooks for analysis, experimentation, and reproducibility

Designed for easy extensibility to test new resampling or modeling approaches

Model Performance Overview
Model	F1 Score	Precision	Recall	PR-AUC
Neural Network (threshold=0.606)	85.25%	90.70%	80.41%	85.70%
Random Forest (threshold=0.364)	84.69%	83.84%	85.57%	85.01%
K-Nearest Neighbors (threshold=0.414)	87.05%	87.50%	86.60%	82.81%
Voting Classifier (threshold=0.283)	86.73%	85.86%	87.63%	87.73%
