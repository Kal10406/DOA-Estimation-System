import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import joblib

# =========================
# LOAD DATA
# =========================
X = np.load("X_data.npy")
y_deg = np.load("y_data.npy")

print("Data loaded:", X.shape, y_deg.shape)

# =========================
# FIX ANGLE (circular representation)
# =========================
y_sin = np.sin(np.deg2rad(y_deg))
y_cos = np.cos(np.deg2rad(y_deg))

y = np.column_stack((y_sin, y_cos))

# =========================
# FEATURE SCALING
# =========================
scaler = StandardScaler()
X = scaler.fit_transform(X)

# =========================
# TRAIN-TEST SPLIT
# =========================
X_train, X_test, y_train, y_test, y_train_deg, y_test_deg = train_test_split(
    X, y, y_deg, test_size=0.2, random_state=42
)
# =========================
# MODEL
# =========================
model = MLPRegressor(
    hidden_layer_sizes=(128, 64, 32),
    activation='relu',
    max_iter=500
)

# =========================
# TRAIN
# =========================
model.fit(X_train, y_train)

# =========================
# PREDICT
# =========================
y_pred = model.predict(X_test)

# =========================
# CONVERT BACK TO ANGLE
# =========================
y_pred_angle = np.rad2deg(np.arctan2(y_pred[:, 0], y_pred[:, 1]))

# =========================
# ERROR
# =========================
mae = mean_absolute_error(y_test_deg, y_pred_angle)

print(f"Mean Absolute Error: {mae:.2f} degrees")

# =========================
# SAVE MODEL + SCALER
# =========================
joblib.dump(model, "doa_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Model saved as doa_model.pkl")
print("Scaler saved as scaler.pkl")