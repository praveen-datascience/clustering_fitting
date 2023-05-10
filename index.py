# -*- coding: utf-8 -*-
"""
Created on Wed May 10 16:34:04 2023

@author: Anthonimuthu Praveenkumar
"""

#required libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from IPython.display import clear_output
from sklearn.cluster import KMeans

def read_my_excel(filename):
    """ This function is used to read excel file and return dataframe """
    excel_result = pd.read_excel(filename)
    return excel_result

#read csv file
df_coemission = read_my_excel("co2_emission_updated.xls")
#check initial dataframe
#print(df_coemission.head())

#selecting required columns
selected_features = ["2000", "2001", "2002", "2003", "2004"]
#empty and null values are cleaned
df_coemission = df_coemission.dropna(subset=selected_features)
#copy cleaned required data
data = df_coemission[selected_features].copy()
#print(data.head())

#kmeans-steps
#step1 : scale the data so that no one column will not dominate other column
#here i used the range from 1 to 10
data = ((data - data.min()) / (data.max() - data.min())) * 10 + 1
data.describe()

#step2 : Initialize random centroids
def choose_random_centroids(data, k):
    """ This function is used to select random centre values for each column """
    centroids = []
    for i in range(k):
        #apply method used to iterate through each column and sample method select random value in each column
        centroid = data.apply(lambda x: float(x.sample()))
        centroids.append(centroid)
    return pd.concat(centroids, axis=1)

#function calling
centroids = choose_random_centroids(data, 3)
#check centre points
#print(centroids)

#step3 : claculate the distance for each data points from centroid using geometry distance formulae
def find_labels(data, centroids):
    """ This function is used to find out index of minimum  value of each centroid column """
    distances = centroids.apply(lambda x: np.sqrt(((data - x) ** 2).sum(axis=1)))
    return distances.idxmin(axis=1)

cluster_labels = find_labels(data, centroids)
#print(cluster_labels)
#cluster_labels.value_counts()

#step4 : update the centroids
def updated_centroids(data, labels, k):
    """ This function is used to update the centroids by taking geometric mean value """
    centroids = data.groupby(cluster_labels).apply(lambda x: np.exp(np.log(x).mean())).T
    return centroids

def plot_clusters(data, labels, centroids, iteration):
    """ This function is used to plot each point under each cluster """
    #pca : used to visualize data in different dimension
    pca = PCA(n_components=2)
    data_2d = pca.fit_transform(data)
    centroids_2d = pca.transform(centroids.T)
    clear_output(wait=True)
    plt.title(f'Iteration {iteration}')
    plt.scatter(x=data_2d[:,0], y=data_2d[:,1], c=labels)
    plt.scatter(x=centroids_2d[:,0], y=centroids_2d[:,1])
    plt.show()

max_iterations = 100
centroid_count = 3

centroids = choose_random_centroids(data, centroid_count)
old_centroids = pd.DataFrame()
iteration = 1

while iteration < max_iterations and not centroids.equals(old_centroids):
    old_centroids = centroids
    labels = find_labels(data, centroids)
    centroids = updated_centroids(data, cluster_labels, centroid_count)
    plot_clusters(data, cluster_labels, centroids, iteration)
    iteration += 1
    
