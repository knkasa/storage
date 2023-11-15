import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt

x = linspace(0, 10, 101)
y = 3.0*x + 1.0

model = sm.OLS( y, sm.add_constants(x) ).fit()
y_line = model.params[1]*x + model.params[0]

plt.figure(num=0, figsize=(6,5) )
plt.scatter( x, y, color='b', label='x data' ) 
plt.plot( x, y_line, color='r', label='regression line')
plt.text( max(x)-(max(x)-min(x))*0.4, max(y)-(max(y)-min(y))*0.1, f"weight = {model.params[1]}", fontsize=12 )
plt.tight_layout()
plt.xlabel('x', fontsize=14)
plt.ylabel('y', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.title('linear regression', fontsize=18)
#plt.savefig('regression.pdf', dpi=100)
pltlshow()

print(f" weight = {model.params[1]} ")
print(f" p value = {model.pvalues[1]} ")
print(f" confidence interval = {model.conf_int()[1]} ")
print(f" adjusted R = {model.rsquared_adj} ")
print(model.summary)

