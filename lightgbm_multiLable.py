import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.multioutput import MultiOutputClassifier
from sklearn.utils import column_or_1d

# Generate some random input data
np.random.seed(42)
num_samples = 1000
num_features = 10
X = np.random.randn(num_samples, num_features)
y1 = np.random.choice(["yes", "no"], num_samples)
y2 = np.random.choice(["yes", "no"], num_samples)
y3 = np.random.choice(["yes", "no"], num_samples)

# Convert the target labels to binary values (0 for "no" and 1 for "yes")
y_binary1 = np.where(y1 == "yes", 1, 0)
y_binary2 = np.where(y2 == "yes", 1, 0)
y_binary3 = np.where(y3 == "yes", 1, 0)

# Combine the three binary target variables into one multi-label target matrix
y_multi = np.column_stack((y_binary1, y_binary2, y_binary3))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_multi, test_size=0.2, random_state=42)

# Create LightGBM model
lgb_model = lgb.LGBMClassifier(
    objective='binary',  # Binary classification task
    metric='binary_logloss',  # Logarithmic loss as the evaluation metric
    num_leaves=31,  # Maximum number of leaves in one tree
    learning_rate=0.05,  # Learning rate
    feature_fraction=0.9,  # Randomly select 90% of features on each iteration
    bagging_fraction=0.8,  # Randomly select 80% of data to train each tree
    bagging_freq=5,  # Perform bagging every 5 iterations
    verbose=0  # Set to 0 to suppress printing messages during training
)

# Create MultiOutputClassifier with LightGBM model
multi_output_lgb_model = MultiOutputClassifier(lgb_model)

# Train the model
multi_output_lgb_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = multi_output_lgb_model.predict(X_test)

# Calculate accuracy for each target column
accuracies = [accuracy_score(y_test[:, i], y_pred[:, i]) for i in range(y_test.shape[1])]
print("Accuracy for Target 1:", accuracies[0])
print("Accuracy for Target 2:", accuracies[1])
print("Accuracy for Target 3:", accuracies[2])
