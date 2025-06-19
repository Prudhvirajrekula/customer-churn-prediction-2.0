import pandas as pd
import numpy as np
import joblib
import json
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, mean_squared_error
from imblearn.over_sampling import SMOTE

print("📥 Loading dataset...")

# Load more rows initially and filter to 100 balanced
df = pd.read_csv("data/model_features.csv", on_bad_lines='skip').dropna().head(1000)

# Check if both classes exist
churn_1 = df[df["is_churned"] == 1.0].head(50)
churn_0 = df[df["is_churned"] == 0.0].head(50)

# Combine to make 100 balanced rows
df = pd.concat([churn_1, churn_0]).reset_index(drop=True)

print("✅ Using", len(df), "balanced rows (50 churned, 50 not churned)")

# Show class balance
print("is_churned")
print(df["is_churned"].value_counts())

# Features and targets (remove customer_id)
feature_cols = ["recency", "monthly_avg", "support_calls", "payment_delay"]
X = df[feature_cols]
y_cls = df["is_churned"]
y_reg = df["monetary"]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Balance classification target
smote = SMOTE(random_state=42)
X_resampled, y_cls_resampled = smote.fit_resample(X_scaled, y_cls)

# Split data
X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(X_resampled, y_cls_resampled, test_size=0.2, random_state=42)
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_scaled, y_reg, test_size=0.2, random_state=42)

print("🧠 Training churn classifier...")
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_cls, y_train_cls)

print("💰 Training LTV regression model...")
reg = LinearRegression()
reg.fit(X_train_reg, y_train_reg)

# Save models and scaler
os.makedirs("models", exist_ok=True)
joblib.dump(clf, "models/model.pkl")
joblib.dump(reg, "models/ltv_regressor.pkl")
joblib.dump(scaler, "models/ltv_scaler.pkl")

# Save performance metrics
metrics = {
    "classification_report": classification_report(y_test_cls, clf.predict(X_test_cls), output_dict=True),
    "ltv_rmse": mean_squared_error(y_test_reg, reg.predict(X_test_reg)) ** 0.5
}

with open("models/ml_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("✅ Models trained and saved successfully.")
