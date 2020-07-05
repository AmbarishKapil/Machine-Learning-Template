# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 20:13:58 2020

@author: Ambarish Kapil
"""

import pandas as pd
from matplotlib import pyplot as plt

df = pd.read_csv('../../../data/Mall_Customers.csv')

X = df.iloc[:,[3,4]].values

# Using dendrogram to find the optimal number of clusters
# =============================================================================
# import scipy.cluster.hierarchy as sch
# dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
# plt.title('Dendrogram')
# plt.xlabel('Customers')
# plt.ylabel('Euclidean distances')
# plt.show()
# =============================================================================

# Applting hierarchical clustering on 5 clusters as determined by the dendrogram
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
y_hc = hc.fit_predict(X)

# Visualising the clusters
plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()