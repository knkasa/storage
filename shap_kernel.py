# Obtaining kernel shap values.

import numpy as np
from itertools import combinations

def calculate_shapley_values(model, instance, baseline):
    """
    Calculate Shapley values manually using Kernel SHAP approximation.

    Args:
        model: A model with a predict function that takes a list of inputs.
        instance: The instance for which Shapley values are calculated.
        baseline: A baseline instance, typically the mean or median input values.

    Returns:
        shapley_values: A list of Shapley values, one for each feature.
    """
    # Number of features
    n_features = len(instance)
    shapley_values = np.zeros(n_features)

    # Iterate over each feature to calculate its Shapley value
    for i in range(n_features):
        shapley_value = 0.0
        # Iterate over all subsets of features excluding the current feature
        for subset_size in range(n_features):
            for subset in combinations([j for j in range(n_features) if j != i], subset_size):
                # Mask for the subset including all features except the current feature
                mask_with_feature = np.array(baseline)
                mask_without_feature = np.array(baseline)

                # Fill the subset with instance values
                for j in subset:
                    mask_with_feature[j] = instance[j]
                    mask_without_feature[j] = instance[j]
                
                # Add the contribution of the current feature to the mask_with_feature
                mask_with_feature[i] = instance[i]
                
                # Predict using the model
                v_with_feature = model.predict(mask_with_feature)
                v_without_feature = model.predict(mask_without_feature)
                
                # Calculate the weight for the Shapley value
                weight = np.math.factorial(len(subset)) * np.math.factorial(n_features - len(subset) - 1) / np.math.factorial(n_features)
                
                # Accumulate the weighted contribution
                shapley_value += weight * (v_with_feature - v_without_feature)
        
        # Store the computed Shapley value for the feature
        shapley_values[i] = shapley_value
    
    return shapley_values

# Example usage
class Model:
    def predict(self, x):
        # Define your model prediction here; for example, a simple sum
        return sum(x)

# Define model, instance, and baseline
model = Model()
instance = [1, 2, 0, 3]  # The input for which you want to calculate Shapley values
baseline = [0, 0, 0, 0]  # Baseline (e.g., zero input values)

# Calculate Shapley values
shapley_values = calculate_shapley_values(model, instance, baseline)
print("Shapley values:", shapley_values)


# From Claude.
'''
import numpy as np
from itertools import combinations
from math import factorial

def shapley_value(model, x, feature_idx):
    """
    Calculate the Shapley value for a single feature of an input sample.
    
    Parameters:
    model (object): A model with a predict() method
    x (numpy.ndarray): Input sample to calculate Shapley value for
    feature_idx (int): Index of the feature to calculate Shapley value for
    
    Returns:
    float: Shapley value for the specified feature
    """
    n = len(x)
    shapley_value = 0
    
    for i in range(n):
        for coalition in combinations(range(n), i):
            if feature_idx in coalition:
                x_coalition = x.copy()
                x_coalition[list(coalition)] = 0
                x_coalition[feature_idx] = x[feature_idx]
                
                x_coalition_without = x.copy()
                x_coalition_without[list(coalition)] = 0
                
                f_coalition = model.predict([x_coalition])[0]
                f_coalition_without = model.predict([x_coalition_without])[0]
                
                shapley_value += (f_coalition - f_coalition_without) * (factorial(i) * factorial(n - i - 1)) / factorial(n)

# Example usage
model = YourModel()
X = np.array([1, 2, 0, 3])
shapley_values = [shapley_value(model, X, i) for i in range(len(X))]
print(shapley_values)
'''

