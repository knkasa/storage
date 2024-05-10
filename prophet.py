import pandas as pd
from prophet import Prophet

# Sample input data (replace with your actual data)
data = {
    'ds': pd.to_datetime(['2020-01-01', '2020-02-01', '2020-03-01', '2020-04-01', '2020-05-01', '2020-06-01', '2020-07-01']),
    'y': [100, 120, 130, 110, 140, 150, 170],
    'exog_1': [5, 7, 8, 4, 6, 9, 10],  # Example exogenous variable 1
    'exog_2': ['low', 'high', 'med', 'low', 'high', 'med', 'low']  # Example exogenous variable 2 (categorical)
}

df = pd.DataFrame(data)

# Create a Prophet model
model = Prophet()

model.add_regressor('exog_1')  # Add numerical exog

df_encoded = pd.get_dummies(df, columns=['exog_2'])
model.fit(df_encoded)  # Fit the model with encoded data

# Define future dates
future_dates = model.make_future_dataframe(periods=3, freq='D')

# Make predictions (replace with dataframe that has exog variables).
forecast = model.predict(future_dates.merge(df[['ds', 'exog_1', 'exog_2']], how='left', on='ds'))

# Print the forecast results (f-string for readability)
print(f"Forecast:\n{forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]}")

# Optional: Plot the forecast (adjust figure size as needed)
model.plot(forecast, figsize=(12, 6))