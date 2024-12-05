# Heart Disease Prediction

A PyTorch-based neural network model to predict the presence of heart disease using the UCI Heart Disease dataset.

## Introduction

This project aims to build a binary classification model to predict heart disease presence based on various medical attributes.

## Features

- Data Cleaning and Preprocessing
- Neural Network with Dropout Regularization
- Training and Evaluation Pipelines
- Visualization of Training Metrics
- Reproducibility with Seed Setting

## Usage

1. **Ensure the dataset is placed in `data/raw/heart_disease_uci.csv` and a virtual environment is created.**

2. **Run the main script:**
    ```bash
    python src/main.py
    ```

## Project Structure


### `src/` Directory

The `src/` directory contains all the source code files essential for data processing, model training, evaluation, and utility functions. 

- **`main.py`**
  
  - **Purpose:** Serves as the entry point of the project. It orchestrates the entire workflow, including data preprocessing, model training, and evaluation.
  
  - **Details:** When executed, it sequentially calls functions from other modules to perform tasks and manage the flow of the project.

- **`data_preprocessing.py`**
  
  - **Purpose:** Handles all data-related operations, including loading, cleaning, encoding, and splitting the dataset.
  
  - **Details:**
    - **Loading Data:** Reads the raw dataset from `data/raw/heart_disease_uci.csv`.
    - **Cleaning:** Addresses missing values, removes unnecessary columns, and ensures data consistency.
    - **Encoding:** Converts categorical variables into numerical formats suitable for modeling.
    - **Splitting:** Divides the dataset into training and testing sets, saving them as `train.csv` and `test.csv` in `data/processed/`.

- **`train.py`**
  
  - **Purpose:** Defines the neural network architecture and manages the training process.
  
  - **Details:**
    - **Model Definition:** Implements a PyTorch-based neural network with layers, activation functions, and dropout regularization.
    - **Training Loop:** Handles the iterative process of training the model over multiple epochs, calculating loss, and updating weights.

- **`evaluate.py`**
  
  - **Purpose:** Evaluates the trained model's performance on the test dataset.
  
  - **Details:**
    - **Loading Predictions:** Generates predictions on the test set and saves them as `predictions/test_predictions.npy`.
    - **Metrics Calculation:** Computes classification metrics including precision, recall, f1-score, and support.
    - **Confusion Matrix & ROC Curve:** Creates visualizations like the confusion matrix (`visualizations/confusion_matrix.png`) and ROC curve (`visualizations/roc_curve.png`) to assess model performance visually.

- **`utils.py`**
  
  - **Purpose:** Contains utility functions that support various parts of the project.
  
  - **Details:**
    - **Seed Setting:** Provides functions to set random seeds across different libraries to ensure reproducibility.
    - **Additional Utilities:** Can be extended to include other helper functions as the project grows.

