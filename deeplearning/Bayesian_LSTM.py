import tensorflow as tf
import tensorflow_probability as tfp

# Set up TensorFlow Probability layers
tfd = tfp.distributions
tfpl = tfp.layers

# Sample data for demonstration
import numpy as np
np.random.seed(42)
time_steps = 100
features = 1

# Generate synthetic time series data
x = np.linspace(0, 100, time_steps)
y = np.sin(x) + 0.1 * np.random.randn(time_steps)

# Prepare data for LSTM
sequence_length = 10
X = []
Y = []

for i in range(len(x) - sequence_length):
    X.append(y[i:i+sequence_length])
    Y.append(y[i+sequence_length])

X = np.array(X).reshape(-1, sequence_length, features)
Y = np.array(Y)

# Split into train and test sets
split = int(0.8 * len(X))
X_train, Y_train = X[:split], Y[:split]
X_test, Y_test = X[split:], Y[split:]

# Define a Bayesian LSTM model
def bayesian_lstm_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(sequence_length, features)),
        tf.keras.layers.LSTM(16, return_sequences=False),
        tfpl.DenseVariational(
            units=1,
            make_posterior_fn=tfpl.util.default_mean_field_normal_fn(),
            make_prior_fn=tfpl.util.default_multivariate_normal_fn,
            activation=None
        )
    ])
    return model

# Negative log-likelihood loss function
def nll(y_true, y_pred):
    return -y_pred.log_prob(y_true)

# Instantiate the model
model = bayesian_lstm_model()

# Compile the model
model.compile(optimizer='adam', loss=nll)

# Train the model
model.fit(
    X_train, Y_train,
    epochs=50,
    batch_size=16,
    validation_data=(X_test, Y_test),
    verbose=1
)

# Make predictions and estimate uncertainty
y_pred_distribution = model(X_test)
y_pred_mean = y_pred_distribution.mean().numpy()
y_pred_stddev = y_pred_distribution.stddev().numpy()

# Plot results
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(Y_test, label='True Values')
plt.plot(y_pred_mean, label='Predicted Mean')
plt.fill_between(
    np.arange(len(y_pred_mean)),
    y_pred_mean - 2 * y_pred_stddev,
    y_pred_mean + 2 * y_pred_stddev,
    color='gray', alpha=0.3, label='Uncertainty (±2 stddev)'
)
plt.legend()
plt.title('Bayesian LSTM Prediction with Uncertainty')
plt.show()
