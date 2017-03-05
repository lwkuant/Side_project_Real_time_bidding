# -*- coding: utf-8 -*-
"""
Real time bidding 
https://www.kaggle.com/zurfer/rtb
"""

### Setup

## Change the working directory to where the data is 
import os 
os.chdir('D:/Dataset/Side_project_Real_time_bidding')

## import required packages 
import pandas as pd 
import numpy as np
import scipy as sp 
import matplotlib.pyplot as plt 
%matplotlib inline 
import seaborn as sns


### Overview of the dataset 
#df = pd.read_csv('biddings.csv.zip', compression='zip'), just for testing 
df = pd.read_csv('biddings.csv')

print(df.shape)
print(df.info())

# Check for NAs
print(np.sum(df.isnull())/len(df))
print(np.argwhere(np.sum(df.isnull())/len(df) > 0)) # there is no column with NAs

# Check the distribution of the target variable 
print(df['convert'].value_counts()) # the difference between classes is huge
print(df['convert'].value_counts().apply(lambda x: x/np.sum(df['convert'].value_counts())*100)) 
# the ratio in %

# Split the data to training and test 
seed = 100
np.random.seed(seed)
from sklearn.model_selection import train_test_split
df_tr, df_test = train_test_split(df, test_size=0.2, stratify=df['convert'], random_state=seed)

# Check the proportion after splitting 
print(np.sum(df_tr['convert'])/len(df_tr)*100)
print(np.sum(df_test['convert'])/len(df_test)*100)
# the proportions are similar 


### EDA on the features
import os 
os.chdir(r'D:\Project\Side_project_Real_time_bidding')
def plot_hist_with_class(col, class_col):
    class_list = list(np.unique(class_col))
    color_list = ['#006400', '#CC0000']
    for cat in class_list:
        sns.distplot(col.values[class_col.values == cat],
                     kde_kws={'alpha':0.9, 'label': cat, 'lw':2}, bins=50,
                    color = dict(zip(class_list, color_list))[cat],
                    hist_kws={'alpha':0.8})
    plt.legend()
           
fig = plt.figure(figsize=[80, 110])
plt.suptitle('Histogram for Each Feature by Convert Type', fontsize=25, y=0.94)
for (i, feature) in enumerate(list(df_tr.columns[:-1]), start=1):
    axes = fig.add_subplot(11, 8, i)
    #axes.tick_params(labelsize=10)
    plot_hist_with_class(df_tr[feature], df_tr['convert'])


    
