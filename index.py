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
selected_features = ["1999", "2004", "2009", "2014", "2019"]
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
centroids = choose_random_centroids(data, 5)
#check centre points
#print(centroids)

#step3 : claculate the distance for each data points from centroid using geometry distance formulae
def find_labels(data, centroids):
    """ This function is used to find out index of minimum  value of each centroid column """
    distances = centroids.apply(lambda x: np.sqrt(((data - x) ** 2).sum(axis=1)))
    return distances.idxmin(axis=1)

cluster_labels = find_labels(data, centroids)
print(cluster_labels)
cluster_labels.value_counts()
