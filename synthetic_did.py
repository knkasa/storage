
# Synthetic Difference in Difference.

import pdb  
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from synthdid.synthdid import Synthdid as sdid
import numpy as np
from synthdid.get_data import quota, california_prop99
pd.options.display.float_format = '{:.4f}'.format

df = california_prop99()

california_estimate = sdid( df, unit="State", time="Year", treatment="treated", outcome="PacksPerCapita").fit().vcov(method='placebo')
print( california_estimate.summary().summary2 )
california_estimate.plot_outcomes()
plt.grid()
plt.show()

print( california_estimate.__dict__.keys() )

omg = california_estimate.weights['omega']
t = california_estimate.weights['lambda']
omg = np.reshape( omg, (38, 1) )

df2 = df.loc[ df.State!='California' ].copy()
df2 = df2.pivot( index='Year', columns='State', values='PacksPerCapita' )

val = np.matmul( df2.loc[ df2.index<1989 ].values, omg )
val2 = np.dot( t, val )
val3 = np.matmul( df2.loc[ df2.index>=1989 ].values, omg )
val4 = val3.sum()/12
delta_control = val4 - val2

df3 = df.loc[ df.State=='California' ].copy()
val5 = df3.loc[ df3.Year<1989, 'PacksPerCapita'].values
val6 = np.dot( t, val5 )
val7 = df3.loc[ df3.Year>=1989, 'PacksPerCapita' ].values
val8 = val7.sum()/12
delta_treat = val8 - val6

# Finaly compare delta_treat and delta_control.  

pdb.set_trace()
