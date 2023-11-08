import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
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
    '''
    Plots a dendrogram from correlation matrix.
    '''

    corr = df.corr()
    num_cols = df.select_dtypes(include=['int64','float64','Int64'])[:].columns

    plt.figure(figsize=(12,5))
    dissimilarity = 1 - abs(corr)
    Z = linkage(squareform(dissimilarity), 
                #'complete', 
                method='ward')

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


def Bivariate_cont_cat(data, cont, cat, category):
    '''
    Bivariate_cont_cat(data, 'vintage', 'churn', 1)
    '''
    #creating 2 samples
    x1 = data[cont][data[cat]==category][:]
    x2 = data[cont][~(data[cat]==category)][:]

    #calculating descriptives
    n1, n2 = x1.shape[0], x2.shape[0]
    m1, m2 = x1.mean(), x2.mean()
    std1, std2 = x1.std(), x2.mean()

    #calculating p-values
    t_p_val = TwoSampT(m1, m2, std1, std2, n1, n2)
    z_p_val = TwoSampZ(m1, m2, std1, std2, n1, n2)

    #table
    table = pd.pivot_table(data=data, values=cont, columns=cat, aggfunc = np.mean)

    #plotting
    plt.figure(figsize = (20,4), dpi=140)

    #barplot
    plt.subplot(1,3,1)
    sns.barplot([str(category),'not {}'.format(category)], [m1, m2])
    plt.ylabel('mean {}'.format(cont))
    plt.xlabel(cat)
    plt.title('t-test p-value = {} \n z-test p-value = {}\n {}'.format(t_p_val,
                                                                z_p_val,
                                                                table))

    # category-wise distribution
    plt.subplot(1,3,2)
    sns.kdeplot(x1, shade= True, color='blue', label = 'churned')
    sns.kdeplot(x2, shade= False, color='green', label = 'not churned', linewidth = 1)
    plt.title('categorical distribution')

    # boxplot
    plt.subplot(1,3,3)
    sns.boxplot(x=cat, y=cont, data=data)
    plt.title('categorical boxplot')



def ping():
    """
    You call ping I print pong.
    """
    print('pong')
