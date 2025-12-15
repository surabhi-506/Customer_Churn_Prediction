# Telco Customer Churn Prediction

## 📌 Project Overview
Customer retention is critical for telecom companies. This project analyzes customer demographics, services, and billing information to predict whether a customer will churn (leave).

## 🛠️ Tech Stack
* **Language:** Python
* **Libraries:** Pandas, NumPy, Scikit-Learn, Imbalanced-learn (SMOTE), XGBoost, Matplotlib, Seaborn.

## 📊 Key Findings
* **High Risk:** Customers with **Month-to-Month contracts** and **Fiber Optic internet** have the highest churn rates.
* **New Customers:** Churn risk is highest within the first **5 months** of tenure.

## 🚀 Model Performance
I trained three models to compare performance:
1.  **Logistic Regression:** High Recall (Good at catching churners).
2.  **Random Forest:** Balanced performance.
3.  **XGBoost:** Best overall Accuracy (~80%).

## 🔧 How to Run
1.  Clone the repository.
2.  Install dependencies: `pip install -r requirements.txt`
3.  Run the analysis: `python Analysis.py`
