import pybaobabdt
import pandas as pd
from scipy.io import arff
from sklearn.tree import DecisionTreeClassifier 

# https://medium.com/top-python-libraries/how-to-visualize-decision-trees-and-random-forest-trees-4ff8eb12570c

data = arff.loadarff('wine-quality-red.arff') # Import dataset
df = pd.DataFrame(data[0])
y = list(df['Class'])
features = list(df.columns)
features.remove('Class')
X = df.loc[:, features]
clf = DecisionTreeClassifier().fit(X, y)
ax = pybaobabdt.drawTree(clf, size=10, dpi=300, features=features)  #Visualize the tree