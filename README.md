# Fraud Detection System

## Overview
This Python project implements a **fraud detection system** using machine learning. The system processes an insurance fraud dataset, trains a **Random Forest Classifier**, and provides a command-line interface for testing fraud detection.

## Features
- Handles missing values
- Encodes categorical data
- Balances dataset using **SMOTE** or **undersampling**
- Trains a **Random Forest Classifier**
- Evaluates model performance using **accuracy, precision, recall, and F1-score**
- Provides a **CLI-based fraud detection interface**

## Dataset
The dataset should be in CSV format and contain a target column labeled **FraudFound_P** with values:
- `0` - Not Fraudulent
- `1` - Fraudulent

## Installation & Dependencies
Ensure you have **Python 3.x** installed and run the following command to install required libraries:
```sh
pip install pandas numpy scikit-learn imbalanced-learn
```

## How to Use
### 1. Run the script
```sh
python fraud_detection.py
```
### 2. Enter details for fraud detection
The script will prompt for various **categorical and numerical inputs**, then predict if the transaction is fraudulent.

## Code Structure
- **Data Preprocessing:** Handles missing values, encodes categorical features, and balances classes.
- **Model Training:** Uses a Random Forest classifier to detect fraud.
- **Model Evaluation:** Evaluates performance metrics.
- **User Interface:** CLI-based fraud detection input.

## Model Performance
The system prints accuracy and a classification report after training. Example output:
```
Accuracy: 94.9%
Classification Report:
               precision    recall  f1-score   support
           0       0.95      0.95      0.95      2906
           1       0.95      0.95      0.95      2886
```

## Future Improvements
- Implement a **GUI or web interface**
- Tune hyperparameters for better accuracy
- Save and load trained models for reusability

## Author
Talha Saeed

## License
This project is open-source and available for modification and improvement.

