import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from lime import lime_tabular
import matplotlib.pyplot as plt

# Generate synthetic tabular data for regression
np.random.seed(42)
num_samples = 1000
num_features = 5

# Creating a random feature matrix
X = np.random.rand(num_samples, num_features)

# Creating a continuous target variable
y = 3 * X[:, 0] + 2 * X[:, 1] - 1.5 * X[:, 2] + np.random.normal(0, 0.5, num_samples)

# Convert the NumPy arrays to a Pandas DataFrame
feature_names = [f"feature_{i}" for i in range(num_features)]
data = pd.DataFrame(X, columns=feature_names)
data['target'] = y

# Split the data into training and testing sets
train_size = int(0.8 * num_samples)
train_data = data.iloc[:train_size, :]
test_data = data.iloc[train_size:, :].drop('target', axis=1)

# Create a random forest regressor
model = RandomForestRegressor(random_state=42)
model.fit(train_data.drop('target', axis=1), train_data['target'])

# Create a LimeTabularExplainer for regression
explainer = lime_tabular.LimeTabularExplainer(train_data.drop('target', axis=1).values,
                                              mode='regression',
                                              feature_names=feature_names)

# Choose an instance for explanation from the test data
instance = test_data.iloc[0]

# Explain the prediction
explanation = explainer.explain_instance(instance.values, model.predict)

# Get the explanation plot as a matplotlib figure
fig = explanation.as_pyplot_figure()

# Save the figure using matplotlib
output_file_path = 'lime_regression_explanation_plot.png'
fig.savefig(output_file_path, bbox_inches='tight')
plt.close(fig)  # Close the figure to release resources

# Now, 'lime_regression_explanation_plot.png' should contain the Lime explanation plot for regression
