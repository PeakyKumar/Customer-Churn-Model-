# Telecom Customer Churn Prediction (Machine Learning)

## Overview
Customer retention is a major challenge in the telecommunications industry. This project focuses on predicting whether a customer is likely to discontinue their service (churn) using historical customer data and machine learning techniques.

The objective is to build an end-to-end predictive system that performs data exploration, preprocessing, feature engineering, model training, and evaluation to identify customers at high risk of churn.

---

## Problem Statement
Telecom companies lose significant revenue due to customer churn. Identifying churn-prone customers in advance allows businesses to design targeted retention strategies and reduce revenue loss.

This project aims to:
- Analyze customer behavior patterns
- Understand key factors influencing churn
- Build a reliable machine learning model for churn prediction

---

## Dataset
The dataset used in this project is the **Telco Customer Churn dataset**, publicly available on Kaggle.

🔗 Dataset link:  
https://www.kaggle.com/blastchar/telco-customer-churn

### Dataset Highlights
The dataset contains:
- Customer account details (tenure, contract type, billing method)
- Service usage information (internet, phone, streaming services)
- Demographic attributes
- Target variable indicating churn status

> Note: The dataset is **not included in this repository**. Please download it separately and place it inside the `data/` directory.

---

## Project Workflow
The project follows a structured machine learning pipeline:

1. **Data Loading & Cleaning**
   - Handling missing and inconsistent values
   - Converting data types for modeling

2. **Exploratory Data Analysis (EDA)**
   - Churn distribution analysis
   - Feature-wise churn trends
   - Statistical summaries and visual insights

3. **Feature Engineering**
   - Derived metrics such as average monthly spend
   - High-value customer indicators
   - Encoding categorical variables

4. **Preprocessing Pipeline**
   - Numerical feature scaling
   - Categorical feature encoding
   - Unified transformation using `ColumnTransformer`

5. **Model Training**
   - Logistic Regression
   - Random Forest
   - Gradient Boosting
   - K-Nearest Neighbors
   - Naive Bayes

6. **Model Evaluation**
   - Accuracy
   - ROC-AUC score
   - Classification report
   - Confusion matrix

7. **Hyperparameter Tuning**
   - GridSearchCV applied to Gradient Boosting
   - Selection of best performing model

---

## Best Performing Model
After experimentation and tuning, **Gradient Boosting Classifier** achieved the strongest overall performance based on ROC-AUC and classification metrics.

The trained model is saved using `joblib` for future inference or deployment.

---

## Tech Stack
- **Programming Language:** Python
- **Libraries:**
  - pandas
  - numpy
  - scikit-learn
  - matplotlib
  - seaborn
  - joblib

---

## Project Structure

