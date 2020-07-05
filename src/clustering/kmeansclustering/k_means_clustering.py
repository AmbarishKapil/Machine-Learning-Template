# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 20:13:24 2020

@author: Ambarish Kapil
"""

import pandas as pd
from matplotlib import pyplot as plt

df = pd.read_csv('../../../data/Mall_Customers.csv')

X = df.iloc[:,1:]

# Encoding categorical values
X = pd.get_dummies(X, columns=['Genre'], drop_first=True)

# Using elbow method to determine optimal number of clusters
# =============================================================================
# from sklearn.cluster import KMeans
# wcss = []
# for i in range(1,11):
#     kmeans = KMeans(n_clusters = i, init = 'k-means++')
#     kmeans.fit(X)
#     wcss.append(kmeans.inertia_)
# plt.plot(range(1, 11), wcss)
# plt.title('The Elbow Method')
# plt.xlabel('Number of clusters')
# plt.ylabel('WCSS')
# plt.show()
# =============================================================================

# Running k means on 5 clusters as determined by the elbow method
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters = 5, init = 'k-means++')
y_kmeans = kmeans.fit_predict(X)

customer_bucket_dict = {'customer_id':[], 'bucket':[]}

for i in range(0, len(y_kmeans), 1):
    customer_bucket_dict['customer_id'].append(df.iloc[i,0])
    customer_bucket_dict['bucket'].append(y_kmeans[i]+1)
    
customer_bucket = pd.DataFrame(customer_bucket_dict)