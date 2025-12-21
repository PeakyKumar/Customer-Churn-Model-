# 📊 Telecom Customer Churn Prediction (Machine Learning)

## 📌 Overview
Customer retention is a critical challenge in the telecommunications industry, where acquiring new customers is significantly more expensive than retaining existing ones.  
This project focuses on predicting **customer churn**—whether a customer is likely to discontinue their service—using **machine learning techniques** applied to historical telecom customer data.

The goal is to build an **end-to-end churn prediction system**, covering data analysis, preprocessing, feature engineering, model training, evaluation, and model selection to identify customers at high risk of churn.

---

## 🎯 Problem Statement
Telecom companies experience substantial revenue loss due to customer churn.  
By identifying churn-prone customers in advance, businesses can design **targeted retention strategies**, improve customer satisfaction, and reduce revenue leakage.

This project aims to:
- Analyze customer behavior and service usage patterns
- Identify key factors influencing churn
- Build and evaluate multiple machine learning models
- Select a reliable model for churn prediction

---

## 📂 Dataset
The dataset used in this project is the **Telco Customer Churn Dataset**, publicly available on Kaggle.

🔗 **Dataset Link:**  
https://www.kaggle.com/blastchar/telco-customer-churn

### Dataset Highlights
The dataset includes:
- Customer account details (tenure, contract type, billing method)
- Service usage information (internet, phone, streaming services)
- Demographic attributes
- Target variable indicating churn status (`Yes` / `No`)

> ⚠️ **Note:**  
> The dataset is not included in this repository.  
> Please download it separately and place it inside the project directory before running the notebook.

---

## 🔄 Project Workflow

### 1️⃣ Data Loading & Cleaning
- Handling missing and inconsistent values
- Correcting data types for modeling

### 2️⃣ Exploratory Data Analysis (EDA)
- Churn distribution analysis
- Feature-wise churn trends
- Statistical summaries and visual insights

### 3️⃣ Feature Engineering
- Derived metrics such as average monthly spend
- Identification of high-value customers
- Encoding of categorical variables

### 4️⃣ Preprocessing Pipeline
- Numerical feature scaling
- Categorical feature encoding
- Unified transformation using `ColumnTransformer`

### 5️⃣ Model Training
The following machine learning models were trained and evaluated:
- Logistic Regression
- Random Forest
- Gradient Boosting
- K-Nearest Neighbors (KNN)
- Naive Bayes

### 6️⃣ Model Evaluation
Models were evaluated using:
- Accuracy
- ROC-AUC score
- Classification report
- Confusion matrix

### 7️⃣ Hyperparameter Tuning
- `GridSearchCV` applied to Gradient Boosting
- Best parameters selected based on ROC-AUC performance

---

## 🏆 Results

### Model Performance Comparison

| Model | Accuracy | ROC-AUC |
|------|---------|---------|
| Logistic Regression | 0.81 | 0.842 |
| Random Forest | 0.79 | 0.825 |
| Naive Bayes | 0.69 | 0.807 |
| Gradient Boosting (Tuned) | **0.81** | **0.845** |

### Best Performing Model
The **Gradient Boosting Classifier**, after hyperparameter tuning, achieved the best overall performance based on ROC-AUC score.

**Best Parameters:**
- Number of estimators: 100  
- Learning rate: 0.05  
- Max depth: 3  

**Test Performance:**
- **Accuracy:** 0.81  
- **ROC-AUC:** 0.845  

**Key Observations:**
- Strong discriminative power for identifying churn-prone customers
- Balanced precision and recall across churn and non-churn classes
- Suitable for real-world customer retention use cases

---

## 🔍 Key Insights
- Customers with **shorter tenure** and **month-to-month contracts** are more likely to churn
- **Monthly charges** and **contract type** are strong predictors of churn
- Gradient Boosting outperformed baseline models in overall classification performance

---

## 🛠️ Tech Stack
**Programming Language:** Python  

**Libraries Used:**
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- joblib

---

## 📁 Repository Structure
- Customer-Churn-Model/
- │
- ├customer_churn_prediction.ipynb
- ├── README.md
- ├── requirements.txt
- ├── .gitignore


---

## 🚀 Future Work
- Address class imbalance using techniques like SMOTE
- Deploy the model using Flask or FastAPI
- Integrate real-time customer data for live churn prediction
- Perform cost-sensitive learning to optimize retention strategies

---

## 📌 Conclusion
This project demonstrates a complete machine learning workflow for telecom customer churn prediction, from raw data analysis to model selection.  
The results highlight the effectiveness of ensemble methods, particularly Gradient Boosting, in predicting customer churn and supporting data-driven business decisions.

---

## 🤝 Acknowledgements
- Kaggle for providing the Telco Customer Churn dataset
- Open-source Python machine learning community

