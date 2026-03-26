import streamlit as st
import numpy as np
import plotly.graph_objects as go
import joblib
import os
port = int(os.environ.get("PORT", 10000))

st.title("DOA Estimation: MUSIC vs ML")

# =========================
# LOAD ML MODEL
# =========================
model = joblib.load("doa_model.pkl")
scaler = joblib.load("scaler.pkl")

# =========================
# PARAMETERS
# =========================
N = st.slider("Number of Antennas", 2, 16, 8)
snr_db = st.slider("SNR (dB)", 0, 30, 20)
theta_true = st.slider("True Angle (°)", -90, 90, 30)

snapshots = 200
d = 0.5

# =========================
# STEERING VECTOR
# =========================
def steering_vector(theta):
    theta_rad = np.deg2rad(theta)
    return np.exp(-1j * 2 * np.pi * d * np.arange(N) * np.sin(theta_rad))

# =========================
# SIGNAL GENERATION
# =========================
s = np.random.randn(snapshots) + 1j*np.random.randn(snapshots)

A = steering_vector(theta_true).reshape(-1,1)

noise_power = 1 / (10**(snr_db/10))
noise = np.sqrt(noise_power) * (
    np.random.randn(N, snapshots) + 1j*np.random.randn(N, snapshots)
)

X = A @ s.reshape(1,-1) + noise

# =========================
# COVARIANCE
# =========================
R = X @ X.conj().T / snapshots

# =========================
# MUSIC
# =========================
eigvals, eigvecs = np.linalg.eig(R)
idx = eigvals.argsort()[::-1]
eigvecs = eigvecs[:, idx]

En = eigvecs[:, 1:]

angles = np.linspace(-90, 90, 360)
P = []

for angle in angles:
    a = steering_vector(angle).reshape(-1,1)
    P.append(1 / np.abs(a.conj().T @ En @ En.conj().T @ a))

P = np.array(P).flatten()

music_angle = float(angles[np.argmax(P)])

# =========================
# ML PREDICTION
# =========================
feature = np.hstack([np.real(R).flatten(), np.imag(R).flatten()])
feature = scaler.transform([feature])

y_pred = model.predict(feature)

ml_angle = float(np.rad2deg(np.arctan2(y_pred[0][0], y_pred[0][1])))

# =========================
# ERRORS
# =========================
music_error = abs(theta_true - music_angle)
ml_error = abs(theta_true - ml_angle)

# =========================
# DISPLAY
# =========================
st.subheader("Results")

st.text(f"True Angle (°): {round(theta_true,2)}")
st.text(f"MUSIC Estimated (°): {round(music_angle,2)}")
st.text(f"ML Estimated (°): {round(ml_angle,2)}")

st.text(f"MUSIC Error (°): {round(music_error,2)}")
st.text(f"ML Error (°): {round(ml_error,2)}")

# =========================
# PLOT
# =========================
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=angles,
    y=10*np.log10(P/np.max(P)),
    name="MUSIC Spectrum"
))

fig.add_vline(x=theta_true, line_dash="dash", line_color="red")
fig.add_vline(x=music_angle, line_dash="dash", line_color="green")
fig.add_vline(x=ml_angle, line_dash="dot", line_color="blue")

fig.update_layout(
    title="DOA Spectrum (Red=True, Green=MUSIC, Blue=ML)",
    xaxis_title="Angle (degrees)",
    yaxis_title="Power (dB)"
)

st.plotly_chart(fig)

# =========================
# INTERPRETATION
# =========================
st.info("""
Red → True angle  
Green → MUSIC estimate  
Blue → ML estimate  

MUSIC uses signal subspace theory  
ML learns mapping from data  
""")
