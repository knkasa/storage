# Obtaining kernel shap values.
# The perturbed value is replaced with the baseline data, and the difference is averaged for all rows.
# https://shap.readthedocs.io/en/latest/generated/shap.KernelExplainer.html

import numpy as np
from itertools import combinations
from sklearn.linear_model import Ridge

def calculate_shapley_values(model, instance, baseline):
    
    n_features = instance.shape[1]
    shapley_values = np.zeros(n_features)

    for i in range(n_features):
        shapley_value = 0.0
        for subset_size in range(n_features):
            for subset in combinations([j for j in range(n_features) if j != i], subset_size):
                diff = 0.0
                for base_row in range(baseline.shape[0]):
                    
                    # Mask for the subset including all features except the current feature
                    mask_with_feature = baseline[base_row][np.newaxis,:].copy()
                    mask_without_feature = baseline[base_row][np.newaxis,:].copy()
                    
                    # Fill the subset with instance values
                    for j in subset:
                        mask_with_feature[0,j] = instance[0,j]
                        mask_without_feature[0,j] = instance[0,j]
                    
                    # Add the contribution of the current feature to the mask_with_feature
                    mask_with_feature[0,i] = instance[0,i]
                    
                    v_with_feature = model.predict(mask_with_feature)
                    v_without_feature = model.predict(mask_without_feature)
                    
                    diff += (v_with_feature - v_without_feature)/baseline.shape[0] # averaged over all rows(the number of baseline rows)
                
                weight = np.math.factorial(len(subset))*np.math.factorial(n_features-len(subset)-1)/np.math.factorial(n_features)
                shapley_value += weight*diff 
        
        shapley_values[i] = shapley_value
    
    return shapley_values

X = np.array([[1, 2, 3, 4],
              [5, 6, 7, 8],
              [9, 10, 11, 12],
              [13, 14, 15, 16]])
y = np.array([10, 20, 30, 40])

model = Ridge(alpha=1.0) 
model.fit(X,y)

instance = np.array([[1, 2, -1, 3]])  # The input for which you want to calculate Shapley values
baseline = np.array([
                    [0,0,0,0],
                    [0,0,0,0],
                    [0,0,0,0],
                    ])  # Baseline (e.g., zero input values)

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

