import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.metrics import silhouette_score



clusters_n = 3
iteration_n = 100

data = pd.read_csv('C:/my_working_env/deeplearning_practice/iris.csv')  


data = data.values
points = data[:,0:4]
points = np.asarray(points).astype('float32')
y = data[:,4]

centroids = tf.slice( tf.random.shuffle(points), [0,0], [clusters_n, -1])

plt.scatter( points[:,0], points[:,1], s=50, alpha=0.5) 
plt.plot( centroids[:,0], centroids[:,1], 'kx', markersize=15)
#plt.show()
plt.close()

# assign closet centroid to each data pointes
def closest_centroids( points, centroids):
	distances = tf.reduce_sum(tf.square(tf.subtract(points, centroids[:,None])), 2)
	assignments = tf.argmin( distances, 0)
	return assignments 
	
def move_centroids(points, closet, centroids):
	return np.array( [ points[closet==k].mean(axis=0) for k in range(centroids.shape[0]) ]  )

for step in range(iteration_n):
	print( step+1, " of ", iteration_n, " finished." )
	closest = closest_centroids(points, centroids)
	centroids = move_centroids(points, closest, centroids)
	sum_distance =  tf.reduce_sum(tf.square(tf.subtract(points, centroids[:,None]))  ) 
	print(" sum of distance = ", sum_distance.numpy() )
	
plt.scatter( points[:,0], points[:,1], c=closest, s=50, alpha=0.5 )
plt.plot( centroids[:,0], centroids[:,1], 'kx', markersize=15 )
plt.show()
	
	
#------------------------------------------------------------------------
# Plot sum of squared errors to determine the # of clusters  
from sklearn.cluster import KMeans, SpectralClustering
sse = []
list_k = list(range(1, 10))
score_list = []
score_cluster = []
for k in list_k:
    km = KMeans(n_clusters=k)
    km.fit(points)
    sse.append(km.inertia_)
    if k>=2:
        score = silhouette_score( points, km.predict(points) ) 
        score_list.append( score )
        score_cluster.append(k)

# Plot  against # of cluster.  
plt.figure(figsize=(6, 6))
plt.plot(list_k, sse, '-o')
plt.xlabel(r'Number of clusters *k*')
plt.ylabel('Sum of squared distance');
plt.show()

plt.plot( score_cluster, score_list, '-o' )
plt.xlabel(r'Number of clusters *k*')
plt.ylabel('score');
plt.show()
