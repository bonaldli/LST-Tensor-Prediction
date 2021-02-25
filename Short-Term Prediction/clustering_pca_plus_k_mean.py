# -*- coding: utf-8 -*-
"""
Created on Wed May  1 16:22:21 2019

@author: zlibn
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

inputdata = U0_nz # the station mode 
X = pd.DataFrame(inputdata)
X.columns = ["Rank{0}".format(i) for i in range(0,inputdata.shape[1])]
from sklearn.preprocessing import StandardScaler
#X_std = StandardScaler().fit_transform(X)
#X_std = pd.DataFrame(X_std)
#X_std.columns = ["Rank{0}".format(i) for i in range(1,51)]
s = A_m_stn # the inflow station code, inflow_stn_namecsv
X = X.set_index(s)
#------------------------------------------------------------------------------
from sklearn.decomposition import PCA
    
pca = PCA(n_components=2)
pca.fit(X)
PCA(copy=True, n_components=2, whiten=False)
#This gives us an object we can use to transform our data by calling transform.
X_2d = pca.transform(X)

X_2d = pd.DataFrame(X_2d)
X_2d.index = X.index
X_2d.columns = ['PC1','PC2']
X_2d.head()

print(pca.explained_variance_ratio_)
#We see that the first PC already explains almost 32% of the variance
    
ax = X_2d.plot(kind='scatter', x='PC2', y='PC1', figsize=(16,8))  
for i, country in enumerate(X_2d.index):
    ax.annotate(
        country, 
        (X_2d.iloc[i].PC2, X_2d.iloc[i].PC1)
    )

#Let's now create a bubble chart, by setting the point size to a value 
#proportional to the mean value for all the years in that particular country
from sklearn.preprocessing import normalize
    
X_2d['stn_mean'] = pd.Series(X.mean(axis=1), index=X_2d.index)
stn_mean_max = X_2d['stn_mean'].max()
stn_mean_min = X_2d['stn_mean'].min()
stn_mean_scaled = (X_2d.stn_mean-stn_mean_min) / stn_mean_max
X_2d['stn_mean_scaled'] = pd.Series(stn_mean_scaled, index=X_2d.index) 
X_2d.head()

X_2d.plot(
    kind='scatter', 
    x='PC2', 
    y='PC1', 
    s=X_2d['stn_mean_scaled']*100, 
    figsize=(16,8))
#PCA Results
from sklearn.cluster import KMeans
    
kmeans = KMeans(n_clusters=5)
clusters = kmeans.fit(X)
cmap = plt.cm.rainbow
import matplotlib.colors
norm = matplotlib.colors.Normalize(vmin=1.5, vmax=4.5)

X_2d['cluster'] = pd.Series(clusters.labels_, index=X_2d.index)

X_2d.plot(
        kind='scatter',
        x='PC2',y='PC1',
        color=cmap(norm(X_2d.cluster.values)), 
        figsize=(16,8))

ax = X_2d.plot(kind='scatter', x='PC2', y='PC1', color=cmap(norm(X_2d.cluster.values)), figsize=(16,8))  
for i, stn in enumerate(X_2d.index):
    ax.annotate(
        stn, 
        (X_2d.iloc[i].PC2, X_2d.iloc[i].PC1),
        size=4
    )
file_path = "C:\\Users\\zlibn\\Desktop"
plt.savefig(file_path, dpi = (300))   

#-----------------------------------------------------------------------------

#Calculating Eigenvecors and eigenvalues of Covariance matrix
mean_vec = np.mean(X, axis=0)
cov_mat = np.cov(X.T) #or X_std
eig_vals, eig_vecs = np.linalg.eig(cov_mat)

# Create a list of (eigenvalue, eigenvector) tuples
eig_pairs = [ (np.abs(eig_vals[i]),eig_vecs[:,i]) for i in range(len(eig_vals))]

# Sort from high to low
eig_pairs.sort(key = lambda x: x[0], reverse= True)

# Calculation of Explained Variance from the eigenvalues
tot = sum(eig_vals)
var_exp = [(i/tot)*100 for i in sorted(eig_vals, reverse=True)] # Individual explained variance
cum_var_exp = np.cumsum(var_exp) # Cumulative explained variance

# PLOT OUT THE EXPLAINED VARIANCES SUPERIMPOSED 
plt.figure(figsize=(10, 5))
plt.bar(range(len(var_exp)), var_exp, alpha=0.3333, align='center', label='individual explained variance', color = 'g')
plt.step(range(len(cum_var_exp)), cum_var_exp, where='mid',label='cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.legend(loc='best')

describe = X.describe()
plt.show()

########################### Cluster with hierachiry clustering######################
import cmath as math
import sys
import scipy.cluster.hierarchy as sch

X_6d_PC = X_6d.iloc[:,0:6]
###### use median as linkage##########
Z = sch.linkage(X_6d_PC, method = 'median')
fig, axes = plt.subplots(1, 1, figsize=(10, 5))
den = sch.dendrogram(Z, labels = X_6d_PC.index, leaf_font_size=10)
plt.title('Dendrogram for the clustering of the dataset on three different varieties of wheat kernels (Kama, Rosa and Canadian)')
plt.xlabel('Station Code')
plt.ylabel('Euclidean distance with dimensions PC1-PC6')
file_path = "C:\\Users\\zlibn\\Desktop\\dendrogram_median"
plt.savefig(file_path, dpi = (300))   
plt.show()

y_pred = getClusterAssignments(X_6d_PC, den)

# from the plot above if we wanna form 5 clusters

from sklearn.cluster import AgglomerativeClustering
###### use ward as linkage##########
mtd = 'weighted'
Z = sch.linkage(X_6d_PC, method = mtd)
fig, axes = plt.subplots(1, 1, figsize=(10, 5))
den = sch.dendrogram(Z, labels = X_6d_PC.index, leaf_font_size=10)
plt.title('Dendrogram for the clustering of stations based on ' + mtd)
plt.xlabel('Station Code')
plt.ylabel('Euclidean distance with dimensions PC1-PC6')
file_path = "C:\\Users\\zlibn\\Desktop\\dendrogram"
plt.savefig(file_path, dpi = (300))   
plt.show()

# create clusters
hc = AgglomerativeClustering(n_clusters=5, affinity = 'euclidean', linkage = 'ward')
# save clusters for chart
y_hc = hc.fit_predict(X_2d_PC)

X_2d['cluster_h_median'] = pd.Series(clusters.labels_, index=X_2d.index)
X_2d['cluster_h_ward'] = pd.Series(y_hc, index=X_2d.index)


X_2d.plot(
        kind='scatter',
        x='PC2',y='PC1',
        color=cmap(norm(X_2d.cluster_h_ward.values)), 
        figsize=(16,8))

ax = X_2d.plot(kind='scatter', x='PC2', y='PC1', color=cmap(norm(X_2d.cluster_h_ward.values)), figsize=(16,8))  
for i, stn in enumerate(X_2d.index):
    ax.annotate(
        stn, 
        (X_2d.iloc[i].PC2, X_2d.iloc[i].PC1),
        size=4
    )
file_path = "C:\\Users\\zlibn\\Desktop\\cluster_h_ward"
plt.savefig(file_path, dpi = (300))   

# get the list of all cluster2
cluster2 = X_2d.index[X_2d['cluster_h_ward'] == 2].tolist()





