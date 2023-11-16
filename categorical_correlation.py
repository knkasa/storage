# https://towardsdatascience.com/the-search-for-categorical-correlation-a1cf7f1888c9
# Cramers V method.
def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x,y)
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2-((k-1)*(r-1))/(n-1))
    rcorr = r-((r-1)**2)/(n-1)
    kcorr = k-((k-1)**2)/(n-1)
    return np.sqrt(phi2corr/min((kcorr-1),(rcorr-1)))
    

#The tetrachoric correlation coefficient is a more accurate measure of the association between two binary variables than the phi coefficient, but it is also more computationally expensive.
#Here is an example of how to calculate the phi coefficient and the tetrachoric correlation coefficient between two binary variables in Python:
#The polychoric correlation coefficient is used when both variables are categorical with more than two labels.
import pandas as pd
from scipy.stats import chi2_contingency, tetrachoric

# Create a DataFrame
df = pd.DataFrame({'X': [0, 1, 1, 0, 1, 0, 1, 0], 'Y': [1, 0, 1, 1, 0, 1, 0, 1]})

# Calculate the phi coefficient
contingency_table = pd.crosstab(df['X'], df['Y'])
chi2, pval, deg_freedom, expected_counts = chi2_contingency(contingency_table)
phi = np.sqrt(chi2 / df.shape[0])

# Calculate the tetrachoric correlation coefficient
tetrachoric_corr = tetrachoric(contingency_table)

# Print the results
print("Phi coefficient:", phi)
print("Tetrachoric correlation coefficient:", tetrachoric_corr)