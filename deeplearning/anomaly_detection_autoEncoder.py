import tensorflow as tf
from tensorflow import keras
import numpy as np

# Load the feature dataset
X = np.loadtxt('features.csv', delimiter=',')
input_dim = X.shape[1]

# Define the autoencoder architecture
autoencoder = keras.Sequential([
    keras.layers.Dense(32, activation='relu', input_shape=(input_dim,)),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(8, activation='relu'),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(input_dim, activation='sigmoid')
])

# Compile the autoencoder
autoencoder.compile(optimizer='adam', loss='mse')

# Train the model
autoencoder.fit(X, X, epochs=50, batch_size=32, validation_split=0.2)

# Perform anomaly detection using the reconstruction error
reconstructions = autoencoder.predict(X)
mse = np.mean(np.power(X - reconstructions, 2), axis=1)

# Set a threshold based on the 95th percentile of reconstruction errors
threshold = np.percentile(mse, 95)
data['anomaly'] = mse > threshold

# Output the number of anomalies detected
print(f"Autoencoder anomalies detected: {sum(data['anomaly'])}")