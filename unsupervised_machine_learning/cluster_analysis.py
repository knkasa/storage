#====================================================================
# Cluster analysis.  
# 
# Created by  Ken Nakatsukasa.  Mar. 25, 2022
#
# k-means, Hierarchical, and DBSCAN methods will be tested.  
#   For Hierarchical, there are four linkage. Check all of them. 
#   For DBSCAN, need to determine eps.  
#
# Run as "python cluster_analysis.py  <config file path>"  Needs config.yml file.
# https://zenn.dev/kinonotofu/articles/a7cb8038bb2433
#====================================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans, SpectralClustering
from envyaml import EnvYAML
import sys
import sqlalchemy as sa
from scipy.cluster.hierarchy import dendrogram, linkage  
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import pdist, squareform
from sklearn import cluster
from sklearn.metrics import silhouette_score
from scipy.interpolate import interp1d


class kmeans:
    def __init__(self, data, params):
    
        self.data = data
        self.params = params
        
        sse = []
        list_k = list(range(1, 10))
        score_list = []
        score_cluster = []
        for k in list_k:
            km = KMeans(n_clusters=k, 
                        max_iter=300, 
                        init='k-means++', 
                        n_init=10)
            km.fit(self.data)
            sse.append(km.inertia_)
            if k>=2:
                score = silhouette_score( self.data, km.predict(self.data) ) 
                score_list.append( score )
                score_cluster.append(k)

        # Plot  against # of cluster.  下げ幅の確認
        plt.figure(figsize=(6, 6))
        plt.plot(list_k, sse, '-o')
        plt.xlabel(r'Number of clusters *k*')
        plt.ylabel('Sum of squared distance')
        plt.title('k-means')
        plt.show()
        plt.close()
        
        plt.plot( score_cluster, score_list, '-o' )
        plt.xlabel(r'Number of clusters *k*')
        plt.ylabel('score');
        plt.show()
        plt.close()

class hierarchical:
    def __init__(self, data, params):
        
        self.data = data
        self.params = params
        
        linkage_list = ['single', 'complete', 'average', 'ward']
        
        for x in linkage_list:
            linked = linkage(self.data, x)  
            labelList = range(1, len(self.data)+1)
            plt.figure(figsize=(10, 7))  
            dendrogram(linked,  
                        orientation='top',
                        labels=labelList,
                        distance_sort='descending',
                        show_leaf_counts=True)
            plt.xlabel('point labels')
            plt.ylabel('The distance and the cluster tress')
            plt.title(x)
            plt.show()
            plt.close()
        
        # Plot dendrogram again to determine the number of clusters.
        model = AgglomerativeClustering(
                    affinity='euclidean',
                    linkage='ward', 
                    distance_threshold=0,
                    n_clusters=None,
                    )
        model = model.fit(self.data)
        plt.title('Euclidean, linkage=ward')  # *******edit here.
        # plot the top three levels of the dendrogram
        self.plot_dendrogram(model, truncate_mode='level', p=3)
        plt.xlabel("Number of points in node (or index of point if no parenthesis).")
        plt.show()
        
    def plot_dendrogram(self, model, **kwargs):
        counts = np.zeros(model.children_.shape[0])
        n_samples = len(model.labels_)
        for i, merge in enumerate(model.children_):
            current_count = 0
            for child_idx in merge:
                if child_idx < n_samples:
                    current_count += 1 
                else:
                    current_count += counts[child_idx - n_samples]
            counts[i] = current_count
        linkage_matrix = np.column_stack([model.children_, model.distances_,counts]).astype(float)
        dendrogram(linkage_matrix, **kwargs)
        
class dbscan:
    def __init__(self, data, params):
    
        self.data = data
        self.params = params
        
        # Determine eps from here.  
        distances = pdist(self.data, metric='euclidean')
        dist_matrix = squareform(distances)
        dist_matrix[dist_matrix == 0] = 100000  # set this number to be something large.  
        min_distance = dist_matrix.min(axis=1)
        plt.hist( min_distance, bins=10, density=False )
        plt.title("Distribution of nearest distance")
        plt.show()
        plt.close()

        distance_list = np.linspace( 0.2, 1, 9 )  # edit here  
        score_list = []
        for x in distance_list:
            scan_data = cluster.DBSCAN(eps=x, min_samples=5, metric='euclidean').fit_predict(self.data)
            print(f"eps = {str(np.round(x, 2))}, Number of Noise: {str(sum(scan_data==-1))}, ({str(len(scan_data))})", sep='')
            print(f"eps = {str(np.round(x, 2))}, Number of cluseters = {str(len(np.unique(scan_data)))} " )
            score = silhouette_score(self.data, scan_data )
            score_list.append(score)
            print("Silhouette score = ", score )
            #self.cluster_plots(self.data[:,0:2], scan_data)
        plt.plot( distance_list, score_list )
        plt.xlabel('eps'); plt.ylabel('score'); 
        plt.show()
        plt.close()
        
    def cluster_plots(self, set1, colours1='gray', title1='Data in 2D'):
        plt.figure(figsize=(4, 4))
        plt.title(title1,fontsize=14)
        plt.xlim(min(set1[:,0]), max(set1[:,0]))
        plt.ylim(min(set1[:,1]), max(set1[:,1]))
        plt.scatter(set1[:, 0], set1[:, 1], s=8, lw=0, c=colours1)
        plt.tight_layout()
        plt.show()
        plt.close()
    
class pnl_interpolation:
    def __init__(self, params):
        
        self.params = params
        sql_engine = params.get("ai_db.read_replica")
        ai_decision_tbl = params.get("ai_db.tables.agent_decisions_tbl")
        sql_query = ( " select agent_id, utc_datetime, realized_pnl, unrealized_pnl from " 
                        + ai_decision_tbl
                        + " where agent_id=29678 order by utc_datetime " )
        
        df = pd.read_sql_query( sql_query, sql_engine)
        pnl = df['realized_pnl'].to_numpy()
        pnl = np.cumsum(pnl)
        x = np.arange(1, len(pnl)+1 )
        
        spline_pnl = interp1d(x, pnl, kind='cubic')
        #import pdb; pdb.set_trace()  
        
        skip = 20  # ******** edite here.  Number of data points to skip.  
        print(f"Number of points (cubic spline)= {str(len(x[0:len(x):skip]))} Original points = {str(len(x))} ")
        plt.plot( x, pnl, label='raw' )
        plt.plot( x[0:len(x):skip], spline_pnl(x)[0:len(x):skip], label='spline')
        plt.title('cumulative pnl')
        plt.ylabel('pips')
        plt.legend( loc=('upper left') )
        plt.show()
        plt.close()


def main():

    dir_path = sys.argv[1]
    os.chdir(dir_path)
    params = EnvYAML("./config.yml" )  # params.get("ai_db.read_replica")   params.get("ai_db.tables.agent_decisions_tbl")

    # Get input data.
    if params.get("input_data.use_iris_data"):
        #data = pnl_interpolation(params)
        data = load_iris().data
    else:
        pnl_interpolation()

    kmeans(data, params)
    hierarchical(data, params)
    dbscan(data, params)

if __name__ == "__main__":
    main()



