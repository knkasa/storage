import numpy as np
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from sklearn.datasets import load_iris
from sklearn.cluster import AgglomerativeClustering

# Example of Hierachical clustering.  First, plot dendrogram to determine the number of clusters, then cluster them.
# https://towardsdatascience.com/machine-learning-algorithms-part-12-hierarchical-agglomerative-clustering-example-in-python-1e18e0075019
# https://data-analysis-stats.jp/%e6%a9%9f%e6%a2%b0%e5%ad%a6%e7%bf%92/scikit-learn%e3%82%92%e7%94%a8%e3%81%84%e3%81%9f%e9%9a%8e%e5%b1%a4%e7%9a%84%e3%82%af%e3%83%a9%e3%82%b9%e3%82%bf%e3%83%aa%e3%83%b3%e3%82%b0-hierarchical-clustering%e3%81%ae%e8%a7%a3%e8%aa%ac/

iris = load_iris()
print( type(iris) )

# iris is a python class object
X = iris.data
Y = iris.target
print(X.shape)

#-----------------------------------------------------------------------------
#Let's plot the dendrogram for our data points, we must use Scipy Library
from scipy.cluster.hierarchy import dendrogram, linkage  
from matplotlib import pyplot as plt

#Determine which linkage? single, complete, average or ward.  
linked = linkage(X, 'single')
labelList = range(1, len(X)+1)

plt.figure(figsize=(10, 7))  
dendrogram(linked,  
            orientation='top',
            labels=labelList,
            distance_sort='descending',
            show_leaf_counts=True)
plt.xlabel('point labels')
plt.ylabel('The distance and the cluster tress')
plt.show()

#---------------------------------------------------------------------------
# Define function for plotting dendrogram
def plot_dendrogram(model, **kwargs):

	# create the counts of samples under each node
	counts = np.zeros(model.children_.shape[0])
	n_samples = len(model.labels_)
	for i, merge in enumerate(model.children_):
		current_count = 0
		for child_idx in merge:
			if child_idx < n_samples:
				current_count += 1 # leaf node
			else:
				current_count += counts[child_idx - n_samples]
		counts[i] = current_count

	linkage_matrix = np.column_stack([model.children_, model.distances_,counts]).astype(float)

	# Plot the corresponding dendrogram
	dendrogram(linkage_matrix, **kwargs)
	
#-------------------------------------------------------------------------
# plot dendrogram again to determine the number of clusters
model = AgglomerativeClustering(
			affinity='euclidean',
			linkage='ward', 
			distance_threshold=0,
			n_clusters=None,
			)

model = model.fit(X)
plt.title('Euclidean, linkage=ward')
# plot the top three levels of the dendrogram
plot_dendrogram(model, truncate_mode='level', p=3)
plt.xlabel("Number of points in node (or index of point if no parenthesis).")
plt.show()


#----------------------------------------------------------------------------
# Finally cluster them.  
#cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='single') #You can easily change the number of clusters to change the horizontal threshold in the dendrogram 
cluster = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward') #You can easily change the number of clusters to change the horizontal threshold in the dendrogram 
cluster_lables=cluster.fit_predict(X) 

plt.scatter(X[:,0], X[:,1], c=cluster.labels_, cmap='rainbow')  
plt.show()

print(" labels for each data points = ", cluster.labels_ )
print(" this works too ", cluster_lables )



