# Calculating correlations between categorical variables.
# https://py-pair.readthedocs.io/quickstart.html

import pandas as pd
import numpy as np
from pypair.association import binary_binary
from pypair.contingency import BinaryTable
import pdb

get_data = lambda x, y, n: [(x, y) for _ in range(n)]
data = get_data(1, 1, 207) + get_data(1, 0, 282) + get_data(0, 1, 231) + get_data(0, 0, 242)

x = [a for a, _ in data]
y = [b for _, b in data]

for m in BinaryTable.measures():
    r = binary_binary(x, y, m) 
    print(f'{r}: {m}')

    
# https://towardsdatascience.com/the-search-for-categorical-correlation-a1cf7f1888c9
# Cramers V method work the best if x, y are both categorical variable with more than 2 labels (binary).
import scipy.stats as ss
def cramers_v(x, y):
    """ 
        calculate Cramers V statistic for categorial-categorial association.
        uses correction from Bergsma and Wicher,
        Journal of the Korean Statistical Society 42 (2013): 323-328
    """
    confusion_matrix = pd.crosstab(x,y)
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2-((k-1)*(r-1))/(n-1))
    rcorr = r-((r-1)**2)/(n-1)
    kcorr = k-((k-1)**2)/(n-1)
    return np.sqrt(phi2corr/min((kcorr-1),(rcorr-1)))
    





