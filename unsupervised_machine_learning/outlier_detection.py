import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.cm as cm
from sklearn import cluster
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import silhouette_score
import pdb

num_rows = 400
num_columns = 10

data = { f'Column_{i+1}': np.random.uniform(0, 1, num_rows) for i in range(num_columns) }
df = pd.DataFrame(data)

distances = pdist( df, metric='euclidean' )
dist_matrix = squareform(distances)
dist_matrix[dist_matrix == 0] = 10e10  # replace 0 distance with something large to avoid unnecessary detection.
min_distance = dist_matrix.min(axis=1)

# plot histogram.
plt.hist( min_distance, bins=100, density=False )
plt.title("Distribution of nearest distance.")
plt.xlabel('distance')
plt.grid()
plt.show()
plt.close()  

distance_list = np.linspace( 0.8, 0.9, 10 )  # edit here  
score_list = []
for x in distance_list:
    scan_data = cluster.DBSCAN( eps=x, min_samples=5, metric='euclidean').fit_predict(df)
    print(f"eps = {np.round(x, 2)}.  Number of Noise: {sum(scan_data==-1)}({len(scan_data)}).  Number of cluseters:{len(np.unique(scan_data))} ", sep='')
    #self.cluster_plots(self.data[:,0:2], scan_data)


tthreashold = 2
outliers = ( abs(df-df.mean()) > threashold*df.std() ).any(axis=1)
df_no_outliers = df[~outliers]
df_outliers = df[outliers]

print(f"Number without outliers:{len(df_no_outliers)}.  Number of outliers:{len(df_outliers)}")


#pdb.set_trace()



