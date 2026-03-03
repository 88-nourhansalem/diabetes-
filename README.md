# Diabetes Risk Prediction Using Machine Learning

This project implements an interpretable machine learning pipeline to predict diabetes risk using health indicators from the CDC's Behavioral Risk Factor Surveillance System (BRFSS).

## Overview

Diabetes is a significant public health challenge. Early identification of high-risk individuals can lead to better health outcomes. This project uses 21 lifestyle, demographic, and clinical health indicators to predict whether an individual is at risk of diabetes or prediabetes.

## Dataset

- **Name:** CDC Diabetes Health Indicators Dataset (BRFSS 2015)
- **Source:** [Kaggle](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset) / [UCI Machine Learning Repository (ID 891)](https://doi.org/10.24432/C53919)
- **Size:** 253,680 respondents × 21 features
- **Target:** `Diabetes_binary` (0 = No Diabetes, 1 = Prediabetes or Diabetes)

## Key Features of the Pipeline

1.  **Exploratory Data Analysis (EDA):** Visualizes class imbalance, feature distributions, and correlations.
2.  **Imbalance Handling:** Uses `BalancedRandomForestClassifier`, XGBoost's `scale_pos_weight`, and LightGBM's `is_unbalance` to handle the minority class.
3.  **Model Comparison:** Evaluates Logistic Regression, Random Forest, XGBoost, and LightGBM using metrics like Recall, F2-Score, ROC-AUC, and AUPRC.
4.  **Threshold Tuning:** Optimizes the classification threshold to maximize F2-Score, prioritizing the minimization of False Negatives in a clinical context.
5.  **Explainability (XAI):** Uses SHAP (SHapley Additive exPlanations) to provide global and local interpretations of model predictions.

## Installation

Ensure you have Python installed, then install the required dependencies:

```bash
pip install numpy pandas matplotlib seaborn scipy scikit-learn imbalanced-learn xgboost lightgbm shap ucimlrepo
```

## How to Run

### Jupyter Notebook

Open `cdc_diabetes_prediction.ipynb` in Jupyter Lab, Jupyter Notebook, or Google Colab.

The notebook includes a portable data loading mechanism:
1. It looks for `diabetes_012_health_indicators_BRFSS2015.csv` locally.
2. If not found, it automatically fetches the dataset from the UCI Machine Learning Repository.

### Python Script (Manual Extraction)

You can extract the code from the notebook cells into a `.py` script and run it directly. The pipeline is designed to be environment-agnostic.

## Authors

CSAI-801 — Group 18 | Winter 2026
