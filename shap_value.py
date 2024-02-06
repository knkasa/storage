import numpy as np
import matplotlib.pyplot as plt

def calculate_shap_values(model, data):
    # Ensure the model is callable (has a predict method)
    assert hasattr(model, 'predict'), "Model must have a predict method"
    
    # Get the number of features
    num_features = data.shape[1]

    # Initialize an array to store SHAP values for each instance and feature
    shap_values = np.zeros((len(data), num_features))

    # Iterate over each instance in the input data
    for i in range(len(data)):
        # Perturb each feature one at a time and calculate the model predictions
        for j in range(num_features):
            perturbed_data = data.copy()
            perturbed_data[i, j] = 0  # Perturb one feature
            perturbed_output = model.predict(perturbed_data[i:i+1, :])
            original_output = model.predict(data[i:i+1, :])
            shap_values[i, j] = (perturbed_output - original_output)

    return shap_values

# Assuming 'final_model' is your scikit-learn model
# Assuming 'new_data' is your input data for which you want SHAP values
shap_values = calculate_shap_values(final_model, new_data)

# Calculate feature importance as the mean absolute SHAP value across instances
feature_importance = np.mean(np.abs(shap_values), axis=0)

# Sort features based on importance
sorted_indices = np.argsort(feature_importance)[::-1]
sorted_features = new_data.columns[sorted_indices]

# Bar plot showing feature importance
plt.figure(figsize=(10, 6))
plt.bar(range(len(sorted_features)), feature_importance[sorted_indices], tick_label=sorted_features)
plt.xlabel('Features')
plt.ylabel('Feature Importance (Mean |SHAP|)')
plt.title('Feature Importance Based on SHAP Values')
plt.xticks(rotation=45, ha="right")
plt.show()
