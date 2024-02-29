import numpy as np
import pandas as pd
from pycaret.datasets import get_data
from pycaret.time_series import TSForecastingExperiment
import pdb

data = get_data("airquality")

# Limiting the data for demonstration purposes.
data = data.iloc[-720:]
data["index"] = pd.to_datetime(data["Date"] + " " + data["Time"])
data.drop(columns=["Date", "Time"], inplace=True)
data.replace(-200, np.nan, inplace=True)
data.set_index("index", inplace=True)

target = "CO(GT)"
exog_vars = ['NOx(GT)', 'PT08.S3(NOx)', 'RH',]
include = [target] + exog_vars
data = data[include]
data.head()

# FH: number of predicitions t+n.  
FH=5  # best to have FH greater than the periodicity found in the pre-analysis.
metric = "mase"
exclude = ["auto_arima", 'arima', "bats", "tbats", "lar_cds_dt", "par_cds_dt"]

exp_auto = TSForecastingExperiment()
exp_auto.setup(
    data=data, 
    target=target, 
    fh=FH, 
    enforce_exogenous=False,
    #index='index',  # index can be replaced with range(0,x) values too.  No need to be datetime format.  
    #ignore_seasonality_test=True,
    numeric_imputation_target="ffill", 
    numeric_imputation_exogenous="ffill",
    #fig_kwargs=global_fig_settings, 
    session_id=42
)

best = exp_auto.compare_models(sort=metric, turbo=False, exclude=exclude)

exp_auto.plot_model(best)
exp_auto.predict_model( best )

final_auto_model = exp_auto.finalize_model(best)

def safe_predict(exp, model):
    """Prediction wrapper for demo purposes."""
    try: 
        future_preds = exp.predict_model(model)
    except ValueError as exception:
        print(exception)
        exo_vars = exp.exogenous_variables
        print(f"{len(exo_vars)} exogenous variables (X) needed in order to make future predictions:\n{exo_vars}")
        
        exog_exps = []
        exog_models = []
        for exog_var in exog_vars:
            exog_exp = TSForecastingExperiment()
            exog_exp.setup(
                data=data[exog_var], fh=FH,
                #index='index',
                numeric_imputation_target="ffill", 
                numeric_imputation_exogenous="ffill",
                #fig_kwargs=global_fig_settings, 
                session_id=42
            )

            # Users can customize how to model future exogenous variables i.e. add
            # more steps and models to potentially get better models at the expense
            # of higher modeling time.
            best = exog_exp.compare_models(
                sort=metric, include=["arima", "ets", "exp_smooth", "theta", "lightgbm_cds_dt",]        
            )
            final_exog_model = exog_exp.finalize_model(best)

            exog_exps.append(exog_exp)
            exog_models.append(final_exog_model)

        # Step 2: Get future predictions for exog variables ----
        future_exog = [
            exog_exp.predict_model(exog_model)
            for exog_exp, exog_model in zip(exog_exps, exog_models)
        ]
        future_exog = pd.concat(future_exog, axis=1)
        future_exog.columns = exog_vars
        
        future_preds = exp.predict_model(model, X=future_exog) 
    
    return future_preds, future_exog   

# Note: For making predictions, the index of the test dataframe 
#      needs to be exactry the same as the test index of the original dataframe.
future_preds, test_data = safe_predict(exp_auto, final_auto_model)
print( exp_auto.predict_model( final_auto_model, X=test_data ) )
test_data['RH'] = 0
test_data['PT08.S3(NOx)'] = 0
print( exp_auto.predict_model( final_auto_model, X=test_data ) )
future_preds.plot()

# save and load
_ = exp_auto.save_model( final_auto_model, "final_slim_model") 
exp_future = TSForecastingExperiment()
final_slim_model = exp_future.load_model("final_slim_model")
temp = pd.DataFrame({'NOx(GT)':[23], 'PT08.S3(NOx)':[3], 'RH':[2]}) 
exp_future.predict_model( final_slim_model, X=temp  )  

import shap
from sklearn.pipeline import Pipeline

if isinstance(final_slim_model, Pipeline):
    final_slim_model = final_slim_model.steps[-1][1]
else:
    final_slim_model = final_slim_model
x = shap.Explainer( final_slim_model )
shap_values = x.shap_values( test_data )

# Get shap values for feature importance.
'''
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
'''

pdb.set_trace()
