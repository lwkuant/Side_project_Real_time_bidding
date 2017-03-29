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

df_tr.index = range(len(df_tr))
df_test.index = range(len(df_test))

# Check the proportion after splitting 
print(np.sum(df_tr['convert'])/len(df_tr)*100)
print(np.sum(df_test['convert'])/len(df_test)*100)
# the proportions are similar 


### EDA on the features

## using visualizations
import os 
os.chdir(r'D:\Project\Side_project_Real_time_bidding')
def plot_hist_with_class(col, class_col):
    class_list = list(np.unique(class_col))
    color_list = ['#006400', '#CC0000']
    for cat in class_list:
        sns.distplot(col.values[class_col.values == cat],
                     kde_kws={'alpha':0.8, 'label': cat, 'lw':2}, bins=50,
                    color = dict(zip(class_list, color_list))[cat],
                    hist_kws={'alpha':0.5})
    plt.legend()
           
fig = plt.figure(figsize=[80, 110])
plt.suptitle('Histogram for Each Feature by Convert Type', fontsize=25, y=0.94)
for (i, feature) in enumerate(list(df_tr.columns[:-1]), start=1):
    axes = fig.add_subplot(11, 8, i)
    #axes.tick_params(labelsize=10)
    plot_hist_with_class(df_tr[feature], df_tr['convert'])
plt.savefig('Hist_each_feature_by_convert_type.png')

## using statistic tests
from scipy.stats import f_oneway

anova_dict = {}

class_list = list(np.unique(df_tr['convert']))

for feature in df_tr.columns[:-1]:
    li_1 = list(df[feature].values[df['convert'].values == class_list[0]])
    li_2 = list(df[feature].values[df['convert'].values == class_list[1]])
    anova_dict[feature] = list(f_oneway(*[li_1, li_2]))
    
# return the outcome of features by its correlation with the target in descending order     
print(sorted(anova_dict, key=anova_dict.get, reverse=True))

# return the pairs that have p-value less than 0.05
good_features_list = [feature for feature in list(anova_dict.keys()) if anova_dict[feature][1]<0.05]
print(good_features_list)
print(sorted(good_features_list, key=lambda x: anova_dict[x][0], reverse=True))                      

df_tr_filtered = df_tr.ix[:, good_features_list]
print(df_tr_filtered.shape)


### Feature engineering 

## using PCA
from sklearn.decomposition import PCA
pca = PCA(0.9)
df_tr_pca = pca.fit_transform(df_tr_filtered)
print(df_tr_pca.shape)
print(pca.explained_variance_ratio_)

df_tr_pca = pd.concat([pd.DataFrame(df_tr_pca), pd.DataFrame(df_tr['convert'])], axis=1)
print(df_tr_pca.shape)
print(df_tr_pca.head())
print(df_tr_pca['convert'].value_counts()/len(df_tr_pca['convert'])*100)


### Modeling 
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score

X = df_tr_pca.ix[:, list(df_tr_pca.columns[:-1])].values
y = df_tr_pca['convert'].values

model = XGBClassifier()

np.random.seed(seed)
outcome = cross_val_score(model, X, y, scoring='f1', cv=5, n_jobs=-1) # spend roughly 5 minutes
print(outcome) # 0, as expected


### Use the Undersampling to make the dataset more balanced (since we have a lot of data)
np.random.seed(seed)

def under_samling(df, ratio=0.5):
    
    index = list(df['convert'][df['convert'] == 0].index)
    
    num_sample = (ratio/(1-ratio))*len(df['convert'][df['convert'] == 1])
    
    index_random = np.random.choice(index, num_sample, replace=False)
    
    df_sample = pd.concat([df.ix[index_random, :], df.ix[(df['convert'] == 1), :]], axis=0)
    
    index_random_all = np.random.choice(list(df_sample.index), len(list(df_sample.index)), replace=False)
    df_sample = df_sample.ix[index_random_all, :]

    return df_sample

df_tr_sample = under_samling(df_tr)
print(df_tr_sample.head())
print(df_tr_sample['convert'].value_counts())

# reindex
df_tr_sample.index = range(len(df_tr_sample))
print(df_tr_sample.head())


### Feature selection
df_tr_sample_pca = pca.transform(df_tr_sample.ix[:, good_features_list])
print(df_tr_sample_pca.shape)
print(pca.explained_variance_ratio_)

df_tr_sample_pca = pd.concat([pd.DataFrame(df_tr_sample_pca), pd.DataFrame(df_tr_sample['convert'])], axis=1)
print(df_tr_sample_pca.shape)
print(df_tr_sample_pca.head())


### Modeling
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score

X = df_tr_sample_pca.ix[:, list(df_tr_sample_pca.columns[:-1])].values
y = df_tr_sample_pca['convert'].values

model = XGBClassifier()

np.random.seed(seed)
outcome = cross_val_score(model, X, y, scoring='f1', cv=5, n_jobs=-1) # spend seconds
print(outcome) # somewhat better

## Precision 
outcome = cross_val_score(model, X, y, scoring='precision', cv=5, n_jobs=-1) # spend seconds
print(outcome) # somewhat better

## Accuracy
outcome = cross_val_score(model, X, y, scoring='accuracy', cv=5, n_jobs=-1) # spend seconds
print(outcome) # somewhat better


