import numpy as np

# =========================
# PARAMETERS
# =========================
N = 8              # antennas
d = 0.5            # spacing
snapshots = 200
num_samples = 2000

# =========================
# STEERING VECTOR
# =========================
def steering_vector(theta):
    theta_rad = np.deg2rad(theta)
    return np.exp(-1j * 2 * np.pi * d * np.arange(N) * np.sin(theta_rad))

# =========================
# GENERATE DATASET
# =========================
X_data = []
y_data = []

for i in range(num_samples):

    # random angle
    theta = np.random.uniform(-90, 90)

    # random SNR
    snr_db = np.random.uniform(0, 30)

    # signal
    s = np.random.randn(snapshots) + 1j*np.random.randn(snapshots)

    # steering
    A = steering_vector(theta).reshape(-1,1)

    # noise
    noise_power = 1 / (10**(snr_db/10))
    noise = np.sqrt(noise_power) * (
        np.random.randn(N, snapshots) + 1j*np.random.randn(N, snapshots)
    )

    # received signal
    X = A @ s.reshape(1,-1) + noise

    # covariance
    R = X @ X.conj().T / snapshots

    # FEATURE (important choice)
    feature = np.hstack([np.real(R).flatten(), np.imag(R).flatten()])

    # store
    X_data.append(feature)
    y_data.append(theta)

# convert to numpy
X_data = np.array(X_data)
y_data = np.array(y_data)

# save dataset
np.save("X_data.npy", X_data)
np.save("y_data.npy", y_data)

print("Dataset generated!")
print("X shape:", X_data.shape)
print("y shape:", y_data.shape)