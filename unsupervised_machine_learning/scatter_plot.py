import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.cm as cm
from sklearn.cluster import KMeans
import umap
import os
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import pdb

#np.random.seed(42)

num_rows = 400
num_columns = 5

data = { f'Column_{i+1}': np.random.uniform(0, 1, num_rows) for i in range(num_columns) }
df = pd.DataFrame(data)

#-------- now clustering ------------------------------

initial_centroids_indices = [0, 1, 2]
initial_centroids = df.iloc[ initial_centroids_indices, : ].values

#scaler = StandardScaler()
#df = pd.DataFrame( scaler.fit_transform(df), columns=df.columns )

num_clusters = 3
kmeans = KMeans( n_clusters=num_clusters, init=initial_centroids, n_init=1, random_state=42 )

cluster_labels = kmeans.fit_predict(df)
df['cluster'] = cluster_labels
#df['cluster'] = df['cluster'].astype('category')

# Pick one data point as separate cluster.
df.loc[ df.index==0, 'cluster' ] = 3
df.loc[ df.index==1, 'cluster' ] = 4

# 2D plot
df_umap = umap.UMAP().fit_transform( df.drop('cluster', axis=1) )
df_umap = pd.DataFrame(df_umap, columns=['UMAP_1', 'UMAP_2'])
df_umap['Cluster'] = cluster_labels

plt.figure(figsize=(6, 5))
sns.scatterplot(x='UMAP_1', y='UMAP_2', hue='Cluster', palette='viridis', data=df_umap, legend='full')
sns.kdeplot(x='UMAP_1', y='UMAP_2', data=df_umap, levels=5, color='black', linewidths=1)
plt.title('UMAP Projection with K-Means Clusters')
#plt.show()
plt.close()

# 3D plot
df_umap3 = umap.UMAP(n_components=3).fit_transform( df.drop('cluster', axis=1) )
df_umap3 = pd.DataFrame(df_umap3, columns=['UMAP_1', 'UMAP_2', 'UMAP_3'])
df_umap3['Cluster'] = cluster_labels

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(df_umap3['UMAP_1'], df_umap3['UMAP_2'], df_umap3['UMAP_3'], c=df_umap3['Cluster'], cmap='viridis', s=50)
#cbar = fig.colorbar(scatter, ax=ax)
plt.title('3D Scatter Plot with K-Means Clusters')
#plt.show()
plt.close()

#------ Now plotting for each columns -------------------

pdf_pages = PdfPages('scatter.pdf')
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
axes = axes.flatten()

cnt = 0
#n_row  = 0
#n_col = 0
total = 0
for i, col1 in enumerate(df.columns.difference(['cluster'])):
    total += 1
    for j, col2 in enumerate(df.columns.difference(['cluster'])):
        
        if i==j:
            kde_ax = sns.kdeplot(df[col1], ax=axes[cnt], color='blue', legend=False)
            kde_ax.set_xlabel(''); kde_ax.set_ylabel('')
            kde_ax.tick_params(bottom=False)
            kde_ax.set_xticks([]); kde_ax.set_yticks([])
            kde_ax.set_title(f'{col1} (Density)', fontsize=8)
            cnt += 1        
        elif j>i:
            #axes[n_row,n_col].scatter(df[col1], df[col2], alpha=0.5, c=df['cluster'], label=df['cluster'] )
            for x in df.cluster.unique():
                temp = df.loc[ df.cluster==x ]
                if x==3 or x==4:
                    axes[cnt].scatter( temp[col1], temp[col2], alpha=0.5, c='k', s=50, label=x )
                else:
                    axes[cnt].scatter( temp[col1], temp[col2], alpha=0.5, s=10, label=x )
            axes[cnt].set_title(f'{col1} vs. {col2}', fontsize=8)
            axes[cnt].set_xticks([]); axes[cnt].set_yticks([])
            #axes[cnt].legend()
            cnt += 1
        
        total += 1

        #n_col += 1
        if cnt==10:
            #plt.tight_layout()
            #plt.show(); 
            pdf_pages.savefig()
            plt.close()
            fig, axes = plt.subplots(2, 5, figsize=(12, 5))
            axes = axes.flatten()
            cnt = 0
            #n_row = 0
            #n_col = 0
        #if cnt==5:
            #n_row += 1
            #n_col = 0

# plot the left over.
#plt.tight_layout()
#plt.show(); 
pdf_pages.savefig()
plt.close()
pdf_pages.close()

#pdb.set_trace()

