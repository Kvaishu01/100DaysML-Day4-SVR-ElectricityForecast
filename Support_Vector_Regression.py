# Day 4 - Support Vector Regression (SVR) for Electricity Consumption Forecasting

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error

# -----------------------------
# Step 1: Generate synthetic dataset
# -----------------------------
np.random.seed(42)

hours = np.arange(0, 24, 0.5)  # every 30 minutes
temperature = 20 + 10 * np.sin(hours / 24 * 2 * np.pi) + np.random.normal(0, 1, len(hours))  # daily temp variation
consumption = (
    200 + 50 * np.sin((hours - 17) / 24 * 2 * np.pi)  # evening peak
    + 0.5 * temperature**2
    + np.random.normal(0, 20, len(hours))
)

df = pd.DataFrame({
    "Hour": hours,
    "Temperature": temperature,
    "Consumption": consumption
})

# -----------------------------
# Step 2: Prepare data
# -----------------------------
X = df[["Hour", "Temperature"]].values
y = df["Consumption"].values

scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

# -----------------------------
# Step 3: Train SVR
# -----------------------------
svr = SVR(kernel='rbf', C=100, epsilon=0.1)
svr.fit(X_scaled, y_scaled)

# -----------------------------
# Step 4: Make predictions
# -----------------------------
y_pred_scaled = svr.predict(X_scaled)
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

# -----------------------------
# Step 5: Evaluation
# -----------------------------
r2 = r2_score(y, y_pred)
mse = mean_squared_error(y, y_pred)

print(f"R² Score: {r2:.3f}")
print(f"MSE: {mse:.2f}")

# -----------------------------
# Step 6: Visualization
# -----------------------------
plt.figure(figsize=(10, 6))
plt.scatter(df["Hour"], df["Consumption"], color="blue", label="Actual Data")
plt.scatter(df["Hour"], y_pred, color="red", label="Predicted (SVR)", alpha=0.6)
plt.xlabel("Hour of Day")
plt.ylabel("Electricity Consumption (kWh)")
plt.title("Electricity Consumption Forecasting using SVR")
plt.legend()
plt.grid(True)
plt.show()
