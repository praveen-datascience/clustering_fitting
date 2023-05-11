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
import sklearn.cluster as cluster
import sklearn.metrics as skmet
from scipy.optimize import curve_fit



def read_my_excel(filename):
    """ This function is used to read excel file and return dataframe """
    excel_result = pd.read_excel(filename)
    return excel_result

#read csv file
df_data = read_my_excel("world_bank_data.xls")
#check initial dataframe
print(df_data.head())

def display_heat(my_data):
    """ This function is used to find out the correlation between columns """
    my_data2 = my_data
    my_data2.set_index('Country Name', inplace=True)
    columns = ['co2_emission', 'agri_machinery' , 'forest_area']
    corr_matrix = my_data2[columns].corr()
    print(corr_matrix)
    sns.heatmap(corr_matrix, cmap='coolwarm', annot=True)
    plt.show()
    
display_heat(df_data)

#selecting required columns
selected_features = ["co2_emission", "agri_machinery"]
#empty and null values are cleaned
df_data = df_data.dropna(subset=selected_features)
#copy cleaned required data
data = df_data[selected_features].copy()
print(data.head())


#Use of curve_fit 
x = data["co2_emission"]
y = data["agri_machinery"]

def linear(x,a,b):
    return a*x+b

def logarithmic(x,a,b):
    return a*np.log(x)+b

constants = curve_fit(logarithmic,x,y)
a_fit = constants[0][0]
b_fit = constants[0][1]
fit = []
for i in x:
    fit.append(logarithmic(i,a_fit,b_fit))

plt.plot(x,y)
plt.plot(x,fit)
plt.grid()
plt.xlabel("Co2 Emission")
plt.ylabel("Agri Machinery")
plt.title("Co2 emission Vs Agri Machinery")
plt.show()



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
centroids = choose_random_centroids(data, 4)
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
    plt.title(f'Co2 Emission & Agri Machinery Clusters \n Iteration {iteration}')
    plt.scatter(x=data_2d[:,0], y=data_2d[:,1], c=labels)
    plt.scatter(x=centroids_2d[:,0], y=centroids_2d[:,1])
    plt.show()

max_iterations = 100
centroid_count = 4
centroids = choose_random_centroids(data, centroid_count)
old_centroids = pd.DataFrame()
iteration = 1

while iteration < max_iterations and not centroids.equals(old_centroids):
    old_centroids = centroids
    labels = find_labels(data, centroids)
    centroids = updated_centroids(data, cluster_labels, centroid_count)
    plot_clusters(data, cluster_labels, centroids, iteration)
    iteration += 1
