# PyCaret Instructions.

# Pycaret automates machine learning task by comparing many models.
#  - Pycaret only works with specific versions.  https://pycaret.gitbook.io/docs/get-started/installation#dependencies
#  - Is is suggested to use IPython with Jupyter notebook.  To use IPython in jupyter, run,
#     python -m ipykernel install --user --name <yourenvname> --display-name <display-name>
#     (finally, choose the kernel IPykernel in jupytet notebook)
#  - https://zenn.dev/murakamixi/articles/9b7f63f6eb79ad
#  - https://aiacademy.jp/media/?p=954

# In case of errors while installing, see https://qiita.com/Laniakea/items/ac3a0207c140565e7277
# If " from pycaret.regression import * " fails with no module found error, uninstall and re-install again.
# If error persists, try  https://stackoverflow.com/questions/68310729/no-module-named-pycaret  and install markupsafe as instructed.

import warnings
warnings.filterwarnings("ignore")  # There may be many warnings when running pycaret.

import pandas as pd
import numpy as np
import pdb

from pycaret.classification import *  # use pycaret.regression for regression, or etc.
from pycaret.datasets import get_data

pdb.set_trace()
dataset = get_data("diamond")

setup(dataset, target="Price", session_id=123,  # session ID is same as random seed.  (If you want to reproduce the model, specify this) 
        categorical_features=["Cut", "Color", "Clarity", "Polish", "Symmetry", "Report"], 
        numeric_features=["Carat Weight"])
        
compare_models()



