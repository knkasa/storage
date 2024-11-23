from sklearn.ensemble import IsolationForest
import numpy as np

# Load the preprocessed features from a CSV file
X = np.loadtxt('features.csv', delimiter=',')

# Initialize and train the Isolation Forest model
model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
model.fit(X)

# Predict anomalies (-1 indicates an anomaly, 1 indicates normal behavior)
predictions = model.predict(X)
anomalies = np.sum(predictions == -1)

print(f"Total anomalies detected: {anomalies}")

# Add predictions to the original dataset for analysis
data['anomaly'] = predictions
print(data[data['anomaly'] == -1].head())