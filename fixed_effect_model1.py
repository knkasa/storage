
import pandas as pd
import scipy.stats as st
import statsmodels.api as sm
import statsmodels.formula.api as smf
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import pdb

# Fixed effect model.  
# https://timeseriesreasoning.com/contents/the-fixed-effects-regression-model-for-panel-data-sets/
# https://medium.com/pew-research-center-decoded/using-fixed-and-random-effects-models-for-panel-data-in-python-a795865736ab

#df_panel = pd.read_csv('wb_data_panel_2ind_7units_1992_2014.csv', header=0)
#df_panel['YEAR'] = df_panel['YEAR'].astype(str)  # convert year to string for one-hot encoding.

data = {
    'Country': ['A', 'B', 'C', 'A', 'B', 'C', 'A', 'B', 'C'],
    'Month': ['Jan', 'Jan', 'Jan', 'Feb', 'Feb', 'Feb', 'Mar', 'Mar', 'Mar'],  
    'X1': [10, 15, 12, 8, 14, 10, 8, 14, 10],
    'X2': [5, 7, 6, 4, 8, 5, 8, 14, 10],
    'Y': [25, 30, 28, 20, 32, 22, 8, 14, 10]
    }
df_panel = pd.DataFrame(data)

unit_col_name='Country'
time_period_col_name='Month'

df_panel[time_period_col_name] = df_panel[time_period_col_name].astype(str)  # convert year to string for one-hot encoding.

unit_names = df_panel[unit_col_name].unique()
unit_times = df_panel[time_period_col_name].unique()

y_var_name = 'Y'
X_var_names = ['X1', 'X2']  

#Create the dummy variables, one for each country
df_dummies = pd.get_dummies(df_panel[unit_col_name], drop_first=False )  # Drop
df_dummies_time = pd.get_dummies(df_panel[time_period_col_name], drop_first=False )

df_panel_with_dummies = df_panel.join(df_dummies)
df_panel_with_dummies = df_panel_with_dummies.join(df_dummies_time)

# Get forumula names for the regression.  Y ~ X1 + X2 + dummy2 + dummy3 + ... 
lsdv_expr = y_var_name + ' ~ '
for i, X_var_name in enumerate(X_var_names):
    if i > 0: # If X has more than 2 columns
        lsdv_expr = lsdv_expr + ' + ' + X_var_name
    else:
        lsdv_expr = lsdv_expr + X_var_name
    i = i + 1

# Add dummy column names to formula.  Exclude the last dummy column for multi-colinear.
for dummy_name in unit_names[:-1]:
    lsdv_expr = lsdv_expr + ' + ' + dummy_name
for dummy_name in unit_times[:-1]:
    lsdv_expr = lsdv_expr + " + "+ dummy_name

print('Regression expression for OLS with dummies:  ' + lsdv_expr)

lsdv_model = smf.ols( formula=lsdv_expr,  data=df_panel_with_dummies)
lsdv_model_results = lsdv_model.fit()
print('')
print('============================== OLSR With Dummies ==============================')
print(lsdv_model_results.summary())
print('LSDV='+str(lsdv_model_results.ssr))

#pdb.set_trace()
exit()
#--------- You may also do as follow. ---------------------------------
df_dummies = pd.get_dummies(df_panel[unit_col_name], drop_first=True )  # exclude the first 
# Adding a constant for the intercept
X = sm.add_constant(df[['X1', 'X2', 'X3', 'Individual_B', 'Individual_C']]) # do not include Individual_A
y = df['Y']

model = sm.OLS(y, X).fit()
#----------------------------------------------------------------------

