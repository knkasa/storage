import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.datasets import fetch_openml
from sklearn.manifold import TSNE

# Dimensionality reduction using TSNE.  Use this for unsupervised clustering.
# MNIST hand written image.  28x28 pixels.  
# https://www.sairablog.com/article/python-tsne-sklearn-matplotlib.html

mnist = fetch_openml('mnist_784', version=1)  # mnist is an image of hand written digit. 784=28x28 pixels.
mnist.target = mnist.target.astype(int)

idx = np.random.permutation(60000)[:10000] # 60000の数字をランダムに置き換えて、10000個抽出してインデックスを作成

X = mnist['data'][idx]
y = mnist['target'][idx]

# use "np.reshape(X, (60000, 784))"  to flatten an image.    
print( np.shape(X) )  # Note the image is flatten. 784=28x28 pixels.

# Use TSNE to reduce dimensionality from 784 to 2
tsne = TSNE(n_components=2, random_state=41)
X_reduced = tsne.fit_transform(X)

# Finally, plot in 2-D.  Different colors corresponds to different labels(integer number)
plt.figure(figsize=(13, 7))
plt.scatter(X_reduced[:, 0], X_reduced[:, 1],
            c=y, cmap='jet',
            s=15, alpha=0.5)
plt.axis('off')
plt.colorbar()
plt.show()

# Finally, use k-means or something to cluster them.  

import pdb;  pdb.set_trace()  