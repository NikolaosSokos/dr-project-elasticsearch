# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 02:11:24 2020

@author: Nikos
"""

from sklearn.cluster import KMeans
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import helper
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable
rows = ['Adventure', 'Animation', 'Children', 'Comedy', 'Fantasy', 'Romance', 'Drama', 'Action', 'Crime', 'Thriller', 'Horror', 'Mystery', 'Sci-Fi', 'Documentary', 'IMAX', 'War', 'Musical', 'Western', 'Film-Noir', '(no genres listed)']
sns.set(context="notebook", palette="Spectral", style = 'darkgrid' ,font_scale = 1.5, color_codes=True)
df = pd.read_pickle("data.pkl")
df=df.astype(float)
df =df.fillna(0)
X = df.iloc[:671].values
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X)
    # inertia method returns wcss for that model
    wcss.append(kmeans.inertia_)
plt.figure(figsize=(10,5))
sns.lineplot(range(1, 11), wcss,marker='o',color='red')
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()
kmeans = KMeans(n_clusters = 3, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(X)

plt.figure(figsize=(15,7))
sns.scatterplot(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], color = 'yellow', label = 'Cluster 1',s=50)
sns.scatterplot(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], color = 'blue', label = 'Cluster 2',s=50)
sns.scatterplot(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], color = 'green', label = 'Cluster 3',s=50)
sns.scatterplot(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], color = 'red', 
                label = 'Centroids',s=300,marker=',')
plt.grid(False)
plt.title('Clusters of customers')
plt.legend()
plt.show()
df["Cluster"] = y_kmeans
df0 = df[df['Cluster'] == 0]
df1 = df[df['Cluster'] == 1]
df2 = df[df['Cluster'] == 2]
def get_mopercluster(df):
    df = df.mean(axis=0) 
    return df
listdfs = [df0,df1,df2]

lstcluster0 = (get_mopercluster(df0))
lstcluster1 = (get_mopercluster(df1))
lstcluster2 = (get_mopercluster(df2))

def replacezeros(df,lst):
    for i in range(len(rows)):
       df[rows[i]].replace(to_replace= float(0),value =lst[i],inplace= True)
              
    return df

result = pd.concat([replacezeros(df0,lstcluster0.tolist())
, replacezeros(df1,lstcluster1.tolist())
,replacezeros(df2,lstcluster2.tolist())
],sort=False).sort_index()
def csv_todfratings():   
    csv_df = pd.read_csv (r'ratings.csv')
    return(csv_df)

ratings = csv_todfratings()

print(result)
