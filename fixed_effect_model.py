
import pandas as pd
import scipy.stats as st
import statsmodels.api as sm
import statsmodels.formula.api as smf
from matplotlib import pyplot as plt
import seaborn as sns
import pdb

# Fixed effect model.  
# https://timeseriesreasoning.com/contents/the-fixed-effects-regression-model-for-panel-data-sets/
# https://medium.com/pew-research-center-decoded/using-fixed-and-random-effects-models-for-panel-data-in-python-a795865736ab


df_panel = pd.read_csv('wb_data_panel_2ind_7units_1992_2014.csv', header=0)

unit_names = ['Belgium', 'CzechRepublic', 'France', 'Ireland', 'Portugal', 'UK', 'USA']
unit_names.sort()

unit_col_name='COUNTRY'
time_period_col_name='YEAR'

y_var_name = 'GDP_PCAP_GWTH_PCNT'
X_var_names = ['GCF_GWTH_PCNT']

colors_master = ['blue', 'red', 'orange', 'lime', 'yellow', 'cyan', 'violet', 'yellow', 'sandybrown', 'silver']
colors = colors_master[:len(unit_names)]


plot_against_X_index=0
sns.scatterplot(x=df_panel[X_var_names[plot_against_X_index]], y=df_panel[y_var_name],
                hue=df_panel[unit_col_name], palette=colors).set(title=
                'Y-o-Y % Change in per-capita GDP versus Y-o-Y % Change in Gross capital formation')
#plt.show()
plt.close()

#Create the dummy variables, one for each country
df_dummies = pd.get_dummies(df_panel[unit_col_name], )

#Join the dummies Dataframe with the panel data set
df_panel_with_dummies = df_panel.join(df_dummies)

#Construct the regression equation. Note that we are leaving out one dummy variable so as to
# avoid perfect multi-colinearity between the 7 dummy variables. The regression model's intercept
# will contain the value of the coefficient for the omitted dummy variable.
# Example equation form  'Y ~ X1 + X2 + EntityEffects' 
lsdv_expr = y_var_name + ' ~ '
i = 0
for X_var_name in X_var_names:
    if i > 0:
        lsdv_expr = lsdv_expr + ' + ' + X_var_name
    else:
        lsdv_expr = lsdv_expr + X_var_name
    i = i + 1
for dummy_name in unit_names[:-1]:
    lsdv_expr = lsdv_expr + ' + ' + dummy_name

print('Regression expression for OLS with dummies= ' + lsdv_expr)


lsdv_model = smf.ols(formula=lsdv_expr, data=df_panel_with_dummies)
lsdv_model_results = lsdv_model.fit()
print('===============================================================================')
print('============================== OLSR With Dummies ==============================')
print(lsdv_model_results.summary())
print('LSDV='+str(lsdv_model_results.ssr))


exit()
#--------- You may also do as follow. ---------------------------------
df_dummies = pd.get_dummies(df_panel[unit_col_name], drop_first=True )  # exclude the first 
# Adding a constant for the intercept
X = sm.add_constant(df[['X1', 'X2', 'X3', 'Individual_B', 'Individual_C']]) # do not include Individual_A
y = df['Y']

model = sm.OLS(y, X).fit()
#----------------------------------------------------------------------

