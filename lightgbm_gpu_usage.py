import lightgbm as lgb
import cudf
import cupy as cp

# Load directly to GPU memory
gdf = cudf.read_csv('massive_data.csv')
X_gpu = gdf.drop(target_col, axis=1).values # CuPy array on GPU

lgb_train = lgb.Dataset(
    data=train_df[feature_columns + categorical_features], 
    label=train_df["Overdue_Days"], 
    weight=abs(train_df["target"]),  # use this when you you want to put weights on high value target.  you may choose any feature col instead of target.
    categorical_feature=categorical_features
    )

# 1. Define the parameters
params = {
    'objective': 'regression',
    'metric': 'rmse',
    'device_type': 'gpu',  # The equivalent of XGBoost's tree_method
    'is_unbalance'=True,  #alternatively use scale_pis_weight
    'min_split_gain':0.1,  # minimum value required in error gain before making further split.

    }

# 2. Train the model
lgb_model = lgb.train(
    params,
    lgb_train,
    num_boost_round=100,
    n_jobs=-1
)


#------- Using dask ----------------------------
import dask.dataframe as dd
import dask_cudf
from dask_ml.model_selection import GridSearchCV
import xgboost.dask as dxgb

# 1. Load data into a Dask DataFrame (on multiple GPUs/nodes)
ddf = dask_cudf.read_csv('massive_data_shards/*.csv')

# 2. Train with Dask-enabled XGBoost
# The magic happens internally, using the same parameters!
dask_model = dxgb.train(
    client, # Your Dask client connection
    params,
    ddf_train,
    num_boost_round=100
)
