

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import optuna

num_data = 1000


# Sample input data (time series)
data = pd.DataFrame({
    'timestamp': pd.date_range(start='2022-01-01', periods=num_data, freq='D'),
    'value': np.sin(np.linspace(0, 100*np.pi, num_data))
})

# Create lag features
data['lag1'] = data['value'].shift(1)
data['lag2'] = data['value'].shift(2)
data['lag3'] = data['value'].shift(3)
data['lag4'] = data['value'].shift(3)
data['lag5'] = data['value'].shift(3)


# Drop rows with missing values
data = data.dropna()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
                                            data[['lag1', 'lag2', 'lag3', 'lag4', 'lag5']], 
                                            data['value'], 
                                            test_size=0.1, 
                                            random_state=42
                                            )

# Create the LightGBM dataset
train_data = lgb.Dataset( X_train, label=y_train )

# Define the objective function for Optuna optimization
def objective(trial):
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting': 'gbdt',
        'num_leaves': trial.suggest_int('num_leaves', 10, 100),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.1),
        'feature_fraction': trial.suggest_uniform('feature_fraction', 0.1, 1.0)  # randamly throw out columns to avoid overfitting.  
    }

    model = lgb.train(params, train_data, num_boost_round=100)
    #model = lgb.train(params, train_data, valid_sets=[valid_data], early_stopping_rounds=10, num_boost_round=100) if you want to set early stopping.  

    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    return rmse

# Run the hyperparameter optimization with Optuna
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)

# Print the best hyperparameters and the best RMSE
best_params = study.best_params
best_rmse = study.best_value

# Train the model
model = lgb.train(best_params, train_data, num_boost_round=100)

# Predict on the test set
y_pred = model.predict(X_test)

for n in range(len(y_pred)):
    print(f" actual = {str(y_test.values[n])},  predicted = {str(y_pred[n])}. ")

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print("Root Mean Squared Error:", rmse)

print("Best Hyperparameters:", best_params)
print("Best RMSE:", best_rmse)

import pdb; pdb.set_trace()  
