# Breast Cancer Classification using Machine Learning

## Overview

This project focuses on classifying breast cancer tumors as benign (non-cancerous) or malignant (cancerous) using machine learning techniques. The Breast Cancer Wisconsin (Diagnostic) dataset from scikit-learn is used for this purpose. The project includes data exploration, preprocessing, model training, and evaluation.

## Table of Contents

- [Introduction](#introduction)
- [Data Collection](#data-collection)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Model Training](#model-training)
- [Model Evaluation](#model-evaluation)
- [Dependencies](#dependencies)

## Introduction

The goal of this project is to build a binary classification model that can accurately predict whether a breast tumor is benign or malignant based on the features extracted from Fine Needle Aspirate (FNA) images.

## Data Collection

The Breast Cancer Wisconsin (Diagnostic) dataset is loaded using `sklearn.datasets.load_breast_cancer()`. This dataset contains 30 features derived from digitized FNA images of breast masses.

## Exploratory Data Analysis (EDA)

- **Feature Distributions**: Histograms are plotted to visualize the distribution of each feature.
- **Bivariate Analysis**: Box plots are used to compare feature distributions for benign and malignant tumors.
- **Correlation Analysis**: A correlation matrix is generated to identify highly correlated features.
- **Principal Component Analysis (PCA)**: PCA is applied to reduce the dimensionality of the dataset and visualize the separation of benign and malignant samples in a 2D scatter plot.

## Data Preprocessing

- **Feature Scaling**: StandardScaler is used to scale the features.
- **Train/Test Split**: The dataset is split into training and testing sets using train_test_split.

## Model Training

The following machine learning models are trained:

- Logistic Regression
- Linear Support Vector Classifier (LinearSVC)
- K-Nearest Neighbors Classifier (KNeighborsClassifier)
- Random Forest Classifier (RandomForestClassifier)
- Support Vector Classifier (SVC)

## Model Evaluation

- Cross-validation is used to evaluate the performance of each model.
- The `evaluate_model` function (defined in `helper_functions/helper_functions.py`) is used to calculate and display various metrics such as accuracy, precision, recall, and F1-score on the test set.
- Confusion matrices are generated to visualize the performance of the models.

## Dependencies

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- IPython

To install the required dependencies, run:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn ipython