import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Creating example data
data = {
    'Individual': ['A', 'B', 'C', 'A', 'B', 'C'],
    'Month': [1, 1, 1, 2, 2, 2],  # Assuming 'Month' represents the time variable
    'X1': [10, 15, 12, 8, 14, 10],
    'X2': [5, 7, 6, 4, 8, 5],
    'Y': [25, 30, 28, 20, 32, 22]
    }

data = {
    'Country': ['A', 'B', 'C', 'A', 'B', 'C', 'A', 'B', 'C'],
    'Month': ['Jan', 'Jan', 'Jan', 'Feb', 'Feb', 'Feb', 'Mar', 'Mar', 'Mar'],  
    'X1': [10, 15, 12, 8, 14, 10, 8, 14, 10],
    'X2': [5, 7, 6, 4, 8, 5, 8, 14, 10],
    'Y': [25, 30, 28, 20, 32, 22, 8, 14, 10]
    }


df = pd.DataFrame(data)

# Using statsmodels formula API with excluding one level for both Individual and Month
model = ols('Y ~ X1 + X2 + C(Country, Treatment("A")) + C(Month, Treatment(1))', data=df).fit()

# Displaying the results
print(model.summary())
print('LSDV='+str(model.ssr))


