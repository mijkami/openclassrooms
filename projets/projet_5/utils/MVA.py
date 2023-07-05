import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#%matplotlib inline

import warnings
warnings.filterwarnings('ignore')
pd.options.mode.chained_assignment = None  # default='warn'


def heatmap_num(df):
    plt.figure(figsize=(8,8))
    sns.set(font_scale=1.5)
    plt.title('Matrice de corrélation de Pearson entre les variables numériques')

    corr = df.corr()
    mask = np.zeros_like(corr, dtype=np.bool_)
    mask[np.triu_indices_from(mask)] = True 

    ax = sns.heatmap(corr, mask=mask, vmin=-1, cmap='coolwarm')
    plt.show()
    sns.set(font_scale=1)


def all_heatmaps(df):
    '''
    Plotting heatmap
        using all methods for all numerical variables 
        (pearson, kendall, spearman)
    '''
    numerical = df.select_dtypes(include=['int64','float64','Int64'])[:]
    plt.figure(figsize=(36,6), dpi=140)
    for j,i in enumerate(['pearson','kendall','spearman']):
        plt.subplot(1,3,j+1)
        correlation = numerical.dropna().corr(method=i)
        sns.heatmap(correlation, linewidth = 2)
        plt.title(i, fontsize=18)


def dendrogram(df):
    from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
    from scipy.spatial.distance import squareform

    corr = df.corr()
    num_cols = df.select_dtypes(include=['int64','float64','Int64'])[:].columns

    plt.figure(figsize=(12,5))
    dissimilarity = 1 - abs(corr)
    Z = linkage(squareform(dissimilarity), 'complete')

    dendrogram(Z, 
            labels=num_cols, 
            orientation='top', 
            leaf_rotation=90
            );


def num_corr(df):
    num_cols = df.select_dtypes(include=['int64','float64','Int64'])[:].columns
    correlation = num_cols.dropna().corr()
    correlation


def pairplot(df, width=10, height=10):
    num_cols = df.select_dtypes(include=['int64','float64','Int64'])[:].columns
    plt.figure(dpi=150)
    g = sns.pairplot(df[num_cols])
    g.fig.set_size_inches(width,height)
    plt.show()

def ping():
    """
    You call ping I print pong.
    """
    print('pong')
