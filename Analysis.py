
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score
from imblearn.over_sampling import SMOTE


sns.set(style="whitegrid")
plt.style.use('seaborn-v0_8-pastel')


df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

print("--- Data Loaded ---")
print(f"Initial Shape: {df.shape}")


df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Drop rows with NaN values (only ~11 rows, safe to drop)
df.dropna(inplace=True)

# FIX 2: Drop customerID
# WHY: IDs are random unique labels. They have no predictive power.
# Keeping them would confuse the model into memorizing IDs instead of patterns.
df.drop(columns=['customerID'], inplace=True)

# Convert Target 'Churn' to binary (1 for Yes, 0 for No) for analysis
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})


print("\n--- Starting Data Visualization ---")

# VISUALIZATION 1: Target Variable Distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='Churn', data=df)
plt.title('Class Distribution (Churn vs Non-Churn)')
plt.show()
print("Insight 1: The dataset is imbalanced (More people stay than leave). We will need SMOTE later.")

# VISUALIZATION 2: Churn by Contract Type
plt.figure(figsize=(8, 5))
sns.countplot(x='Contract', hue='Churn', data=df)
plt.title('Churn Rate by Contract Type')
plt.show()
print("Insight 2: Month-to-month customers have a VERY high churn rate compared to 1-2 year contracts.")

# VISUALIZATION 3: Churn by Internet Service
plt.figure(figsize=(8, 5))
sns.countplot(x='InternetService', hue='Churn', data=df)
plt.title('Churn Rate by Internet Service')
plt.show()
print("Insight 3: Fiber Optic customers churn more frequently than DSL (Likely due to high price or service issues).")

# VISUALIZATION 4: Numerical Distributions (Tenure)

plt.figure(figsize=(10, 5))
sns.kdeplot(df[df['Churn'] == 0]['tenure'], color='green', shade=True, label='Not Churn')
sns.kdeplot(df[df['Churn'] == 1]['tenure'], color='red', shade=True, label='Churn')
plt.title('Tenure Distribution: Churn vs No Churn')
plt.legend()
plt.show()
print("Insight 4: Churn spikes at low tenure (New customers are at highest risk).")

# VISUALIZATION 5: Correlation Matrix

plt.figure(figsize=(12, 8))
# Select only numeric columns for correlation
numeric_df = df.select_dtypes(include=[np.number])
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()
print(
    "Insight 5: 'Tenure' has a negative correlation (Longer stay = Less Churn). 'MonthlyCharges' has a positive correlation.")


# A. One-Hot Encoding for Categorical Variables
# We use drop_first=True to avoid multicollinearity (The "Dummy Variable Trap")
categorical_cols = df.select_dtypes(include=['object']).columns
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)


X = df_encoded.drop('Churn', axis=1)
y = df_encoded['Churn']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# D. Feature Scaling

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)  # Transform test data using Train statistics (Avoids Leakage)

# E. SMOTE (Synthetic Minority Over-sampling Technique)
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print(f"\nTraining set after SMOTE: {X_train_resampled.shape}")


print("\n--- Training Models ---")

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
}

results = {}

for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train_resampled, y_train_resampled)
    y_pred = model.predict(X_test)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    results[name] = {"Accuracy": acc, "Recall": rec}

    print(f"\n--- {name} Results ---")
    print(classification_report(y_test, y_pred))


# 6. CONCLUSION & VERDICT

print("\n================ FINAL VERDICT ================")
print(f"{'Model':<20} | {'Accuracy':<10} | {'Recall (Churn Catch Rate)':<10}")
print("-" * 50)
for name, metrics in results.items():
    print(f"{name:<20} | {metrics['Accuracy']:.4f}     | {metrics['Recall']:.4f}")

print("\nANALYSIS:")
print("1. Accuracy tells us overall correctness, but Recall is MORE important here.")
print("   Why? We want to catch CHURNERS. Missing a churner (False Negative) costs money.")
print("2. Logistic Regression often has high recall but lower accuracy (it guesses 'Yes' a lot).")
print("3. Random Forest/XGBoost usually provide the best balance.")
print("   - If you want to be aggressive and catch everyone: Pick the model with highest RECALL.")
print("   - If you want a balanced approach: Pick the model with highest F1-SCORE.")