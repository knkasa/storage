# PyCaret Instructions.  https://pycaret.gitbook.io/docs/

# Pycaret automates machine learning task by comparing many models.
#  - Pycaret only works with specific versions (3.8, 3.9, 3.10).  https://pycaret.gitbook.io/docs/get-started/installation#dependencies
#  - If using jupyter notebook(you don't have to), you want to make the environment available in jupyter notebook.
#     python -m ipykernel install --user --name <yourenvname> --display-name <display-name>
#     (finally, choose you kernel of your choice)
#  - https://zenn.dev/murakamixi/articles/9b7f63f6eb79ad
#  - https://aiacademy.jp/media/?p=954

# In case of errors while installing, see https://qiita.com/Laniakea/items/ac3a0207c140565e7277, or try pip install --ignore-installed pycaret
# If "from pycaret.regression import * " fails with no module found error, uninstall and re-install again.
# If error persists, try  https://stackoverflow.com/questions/68310729/no-module-named-pycaret  and install markupsafe(no need to set version) as instructed. After this, you may need to change numpy, scipy, etc versions.

import warnings
warnings.filterwarnings("ignore")  # There may be many warnings when running pycaret.

import pandas as pd
import numpy as np
import pdb

from pycaret.classification import *  # use pycaret.regression for regression, clustering, or etc.
from pycaret.datasets import get_data

dataset = get_data("diamond")

setup(dataset, target="Price", session_id=123,  # session ID is same as random seed.  (If you want to reproduce the model, specify this) 
        categorical_features=["Cut", "Color", "Clarity", "Polish", "Symmetry", "Report"], 
        numeric_features=["Carat Weight"])

# ordinal encoding
setup(data = employee, target = 'left', ordinal_features = {'salary' : ['low', 'medium', 'high']})

# This will give a summary of the model performance.  
best_model = compare_models()  # fold=5, if you want cross validation.
res_models = pull()  # get the summary table.

# Hyperparameter tuning using the best model.
tuned_model = tune_model(best_model)

# It is generally advised to train from scratch.  Run below.
lr_model = create_model('lr')
tuned_lr_model = tune_model(lr_model)
evaluate_model(tuned_lr_model)    # display the training stats.
# final_lr_model = finalize_model(tuned_lr_model)   # finalize training using entire dataset.  You don't need to do this.

# save the model in local.
save_model(final_lr_model, 'final_lr_model')

# Make predictions on new data.
predictions = predict_model(final_lr_model, data=new_data)

# Get feature importance as a pandas DataFrame
feature_importance = get_model(lr_model_trained, 'feature_importance')

# Result plotting.  
# https://qiita.com/ground0state/items/57e565b23770e5a323e9
plot_model(best, plot = 'residuals')  # differences between true values and predictions.  R square is also shown. R>0.7 is consider good(chatgpt).
plot_model(best, plot = 'error')  # x-axis(true value) y-axis(prediction).  The blue points and "best fit" line needs to be close to "indentity" line.
plot_model(best, plot = 'feature')
interpret_model(best, plot = 'summary')
