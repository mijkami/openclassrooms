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


def dendrogram(df):
    from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
    from scipy.spatial.distance import squareform

    corr = df.corr()
    num_cols = df.columns[df.dtypes == 'float64']

    plt.figure(figsize=(12,5))
    dissimilarity = 1 - abs(corr)
    Z = linkage(squareform(dissimilarity), 'complete')

    dendrogram(Z, 
            labels=num_cols, 
            orientation='top', 
            leaf_rotation=90
            );

def ping():
    """
    You call ping I print pong.
    """
    print('pong')
