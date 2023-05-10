# -*- coding: utf-8 -*-
"""
Created on Wed May 10 16:34:04 2023

@author: Anthonimuthu Praveenkumar
"""

#required libraries
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
print(df_coemission.head())
