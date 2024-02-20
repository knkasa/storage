# PyCaret Instructions.  https://pycaret.gitbook.io/docs/
# You'll also need to install catboost, xgboost, shap.  
# Learning curve interpretation.  https://www.dataquest.io/blog/learning-curves-machine-learning/
#  1. Both scores increase with training size: This indicates the model is learning from the data.
#  2. Validation score plateaus before training score: This shows good generalization, meaning the model performs well on unseen data without overfitting to the training data.
#  3. Small gap between the lines: This suggests the model learns effectively without excessive memorization of training data.

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
from pycaret.regression import *  # use pycaret.regression for regression, clustering, classification, timeseries, or etc.
from pycaret.datasets import get_data

dataset = get_data("diamond")

model_setup = setup(
    data=dataset, 
    target="Price", 
    session_id=123,  # session ID is same as random seed.  (If you want to reproduce the model, specify this) 
    train_size=0.7,
    test_data=None,
    
    # Categorical options.
    categorical_features=["Cut", "Color", "Clarity", "Polish", "Symmetry", "Report"], 
    numeric_features=["Carat Weight"],
    ordinal_features=None, #{'salary' : ['low', 'medium', 'high']},
    date_features=None,
    text_features=None,
    max_encoding_ohe=25,  # max one-hot encoding dimension.
    ignore_features=None,

    # Feature engineering.
    preprocess=True,    # applies normalization, data cleaning, encoding.  
    remove_multicollinearity=False,
    multicollinearity_threshold=0.9,  # used if remove_multicollinearity=True
    remove_outliers=False,
    outliers_method="iforest",  # used if remove_outliers=True
    outliers_threshold=0.05,    # used if remove_outliers=True
    transformation=False,    # applies various preprocessing to input data including pca and polynomial transformation. Method is defined in transformation_method.
    transformation_method='yeo-johnson',  # others are 'quantile'.  
    normalize=False,
    normalize_method='zscore',   # minmax, robust, maxabs are available.
    
    # Feature selections.  
    pca=False,
    pca_method='linear',   # used if pca=True.  others are 'kernel' 'incremental'
    pca_components=None,  # number of features to keep using pca.  
    feature_selection=False,   # set it True if you want to throw out unimportant features.
    feature_selection_method='classic',  # used if feature_selection=True.  'univariate' 'sequential'
    feature_selection_estimator='lightgbm',
    n_features_to_select=0.2,   #  fraction of features to keep. It can be integer as well. enabled if feature_selection=True.

    # data preparation.
    data_split_shuffle=True,  
    data_split_stratify=False,  # when enabled, it will split test and training data with the right proportion of target variable.  Needed for classification.
    fold = 4, # number of split to training data.  
    fold_strategy='kfold',  # others are 'groupkfold', 'timeseries' 'stratifiedkfold'
    fold_shuffle=False,   #
    n_jobs=-1 ,  # number of parallel processes.
    use_gpu=False, 

    system_log=False, 
    )


# This will give a summary of the model performance.  
best_model = compare_models(
    include=None,  # List of models to compare.  None means compare all models.
    exclude=['lightgbm'],   # List of models to exclude.
    n_select=1,   # number of top models to return.
    turbo=False,   # if set true, it will speed up the hyperparameter tuning. 
    sort='R2',   # metric used to rank models.  
    errors='ignore',  
    verbose=True,
    )  # fold=5, if you want cross validation.

res_models = pull()  # get the summary table in pandas dataframe.

# It is generally advised to train from scratch.  Run below.
model = create_model(best_model)  # you may choose other models like 'lr' 'lightgbm' and etc.
model = tune_model(
    estimator=model,
    n_iter=10,   # number of iteration in the grid search.
    optimize='R2',   # metric used to evaluate models.
    search_library='optuna',   # library used for tuning hyperparameters.  'optuna'(you'll need pip install optuna.)  'scikit-optimize'(pip install scikit-optimize.)
    search_algorithm='tpe',   # search algorithm. 'scikit-learn', 'grid', 'bayesian'(pip install scikit-optimize), 'tpe'(for optuna) ...
    early_stopping='Hyperband',   # 'ash', 'Hyperband', 'median' ... Ignored if search_library is scikit-learn.  choose scikit-optimize, optuna, ...
    early_stopping_max_iters=10,
    verbose=True,
    )


evaluate_model(model)      # display the training stats.
# final_lr_model = finalize_model(tuned_lr_model)   # finalize training using entire dataset.  You don't need to do this.

# save the model in local.
save_model(model, 'final_lr_model')

# Make predictions on new data.
predictions = predict_model(model, data=new_data)

# Get feature importance as a pandas DataFrame
feature_importance = get_model(dataset, 'feature_importance')
feature_names = model.feature_names_
feature_importance_values = model.feature_importances_

# Get parameters of the model like depth, learning_rate.
model.get_params()
model.get_all_params()

# This may work for catboost.  Plot learning curve.
# Assuming your validation data is in dataset.loc[:100]
import catboost as cb
pool = cb.Pool(dataset.loc[:100])  # Create a Pool object
val_loss = model.eval_metrics(pool, metrics=["RMSE"])
rmse_value = val_loss["RMSE"]

#================ Plotting =============================================================
# Result plotting.  
# https://qiita.com/ground0state/items/57e565b23770e5a323e9
plot_model(
    model,     # differences between true values and predictions.  R square is also shown. R>0.7 is consider good(chatgpt).
    plot='residuals', 
    scale=1,
    save=False,   # save figure.
    ) 
plot_model(model, plot='error')  # x-axis(true value) y-axis(prediction).  The blue points and "best fit" line needs to be close to "indentity" line.
plot_model(model, plot='feature')  # shows top 10 feature importance.  Choose 'feature_all' to plot all.
interpret_model(model, plot='summary')

plot_model(model, plot='pipeline')   # shows drawing of the preprocesssing pipeline.
plot_model(model, plot='cooks')   # Cooks distance plot.
plot_model(model, plot='rfe')   # shows performance vs. # of features.
plot_model(model, plot='learning')
plot_model(model, plot='vc')   # validation curve.
plot_model(model, plot='manifold')  # manifold learning curve.
plot_model(model, plot='tree')   # decision tree
plot_model(model, plot='parameter')   # hyper parameters. 
