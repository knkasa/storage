import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.metrics import silhouette_score
from sklearn import cluster, datasets, mixture
from sklearn.neighbors import kneighbors_graph

#======== DBSCAN clustering ======================

#半径以内に点がいくつあるかでその領域をクラスタとして判断します。近傍の密度がある閾値を超えている限り，クラスタを成長させ続けます。半径以内に近く点がない点はノイズになります。
#https://data-analysis-stats.jp/python/dbscan%E3%82%AF%E3%83%A9%E3%82%B9%E3%82%BF%E3%83%BC%E3%81%AE%E8%A7%A3%E8%AA%AC%E3%81%A8%E5%AE%9F%E9%A8%93/

# HDBSCAN is better approach.
https://hdbscan.readthedocs.io/en/latest/how_hdbscan_works.html
https://qiita.com/ozaki_inu/items/45fb17cd3596a64ed489

# 塊のデータセット
dataset1 = datasets.make_blobs(n_samples=1000, random_state=10, centers=6, cluster_std=1.2)[0]
# 月のデータセット
dataset2 = datasets.make_moons(n_samples=1000, noise=.05)[0]

#---------------------------------------------------------------------------------------------------------------
# グラフ作成
def cluster_plots(set1, set2, colours1='gray', colours2='gray', title1='Dataset 1', title2='Dataset 2'):
	fig,(ax1,ax2) = plt.subplots(1, 2)
	fig.set_size_inches(6, 3)

	ax1.set_title(title1,fontsize=14)
	ax1.set_xlim(min(set1[:,0]), max(set1[:,0]))
	ax1.set_ylim(min(set1[:,1]), max(set1[:,1]))
	ax1.scatter(set1[:, 0], set1[:, 1],s=8,lw=0,c= colours1)

	ax2.set_title(title2,fontsize=14)
	ax2.set_xlim(min(set2[:,0]), max(set2[:,0]))
	ax2.set_ylim(min(set2[:,1]), max(set2[:,1]))
	ax2.scatter(set2[:, 0], set2[:, 1],s=8,lw=0,c=colours2)

	fig.tight_layout()
	plt.show()

# First plot to see what it looks like.
print("First plot the dataset")
cluster_plots(dataset1, dataset2)

#-----------------------------------------------------------------------------------------------------------
# k-mean++クラスタリング
start_time = time.time()

kmeans_dataset1 = cluster.KMeans(n_clusters=4, max_iter=300, init='k-means++', n_init=10).fit_predict(dataset1)
kmeans_dataset2 = cluster.KMeans(n_clusters=2, max_iter=300, init='k-means++', n_init=10).fit_predict(dataset2)
print("--- %s seconds ---" % (time.time() - start_time))
print('Dataset1')
print(*["(k-means) Number of points in cluster "+str(i)+": "+ str(sum(kmeans_dataset1==i)) for i in range(4)], sep='\n')
cluster_plots(dataset1, dataset2, kmeans_dataset1, kmeans_dataset2)

#-----------------------------------------------------------------------------------------------------------
# DBSCANクラスタリングを作成

from scipy.spatial.distance import pdist, squareform
distances = pdist(dataset1, metric='euclidean')
dist_matrix = squareform(distances)
dist_matrix[dist_matrix == 0] = 100000  # set this number to be something large.  
min_distance = dist_matrix.min(axis=1)
plt.hist( min_distance, bins=10, density=False )
plt.title("Histogram of nearest distance for dataset1")
plt.show()

start_time = time.time()

# Notice that DBSCAN doesn't need to specify the number of clusters.  
# eps=minimum distance (between neighborhood) that is needed to be considered for a cluster.
# min_samples= minimum # of data points (within eps) that is needed to be considered a cluster.
dbscan_dataset1 = cluster.DBSCAN(eps=1, min_samples=5, metric='euclidean').fit_predict(dataset1)
dbscan_dataset2 = cluster.DBSCAN(eps=1, min_samples=5, metric='euclidean').fit_predict(dataset2)

# noise points are assigned -1
print("--- %s seconds ---" % (time.time() - start_time))
print('Dataset1:')
print("Number of Noise Points: ",sum(dbscan_dataset1==-1)," (",len(dbscan_dataset1),")",sep='')
print('Dataset2:')
print("Number of Noise Points: ",sum(dbscan_dataset2==-1)," (",len(dbscan_dataset2),")",sep='')
dbscan_dataset2 = cluster.DBSCAN(eps=0.1, min_samples=5, metric='euclidean').fit_predict(dataset2)  # res=DBSCAN.fit(x), res.labels_  also work.    
cluster_plots(dataset1, dataset2, dbscan_dataset1, dbscan_dataset2)

#import pdb;  pdb.set_trace()
