import pandas as pd
import json
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix


# Load dataset
df = pd.read_csv("train.csv")

# Separate features and target
X = df.drop("price_range", axis=1)
y = df["price_range"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Scale features
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = LogisticRegression(max_iter=1000)

model.fit(X_train_scaled, y_train)

# Prediction
y_pred = model.predict(X_test_scaled)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)

print("Accuracy:", accuracy)
print("Confusion Matrix:")
print(cm)

# Save model and scaler
joblib.dump(
    {
        "model": model,
        "scaler": scaler
    },
    "mobile_price_model.pkl"
)

# Save metrics
metrics = {
    "accuracy": float(accuracy),
    "confusion_matrix": cm.tolist()
}

with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("Model saved successfully.")
print("Metrics saved successfully.")
