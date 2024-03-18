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

def boxplot_nums(df, cols_lists, log=False):
    sns.set(rc={'figure.figsize':(11.7,8.27)})
    col_lists_len = len(cols_lists)
    all_lists_len = []
    iter = 0
    for i in cols_lists:
        all_lists_len.append(len(i))
    
    fig, ax  = plt.subplots(1,col_lists_len, gridspec_kw={'width_ratios': all_lists_len})
    
    for i in cols_lists:
        if log==True:
            sns.boxplot(data=np.log(df[i]), ax=ax[iter]
                        , showmeans=True
                        , meanprops={"marker":"s","markerfacecolor":"white", "markeredgecolor":"green"}
                       )
        else:
            sns.boxplot(data=df[i], ax=ax[iter]
                        , showmeans=True
                        , meanprops={"marker":"s","markerfacecolor":"white", "markeredgecolor":"green"}
                       )
        ax[iter].set_xticklabels(ax[iter].get_xticklabels(),rotation=55)
        iter+=1
    
    plt.show()

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

def show_pearson(df, num_cols):
    sns.set_theme(style="white")
    # Compute the correlation matrix
    corr = df[num_cols].corr()
    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))
    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})\
       .set_title('Matrice de corrélation de Pearson entre les variables numériques')

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


def dendrogram(df, ward=False):
    '''
    Plots a dendrogram from correlation matrix.
    '''
    corr = df.corr()
    num_cols = df.select_dtypes(include=['int64','float64','Int64'])[:].columns

    plt.figure(figsize=(12,5))
    dissimilarity = 1 - abs(corr)    
    
    if ward==True:
        Z = linkage(squareform(dissimilarity), method='ward')
    else:
        Z = linkage(squareform(dissimilarity), 'complete')

    dendrogram(Z, 
            labels=num_cols, 
            orientation='top', 
            leaf_rotation=75
            );
    plt.suptitle('CHA : Classification Hiérarchique Ascendante', fontsize=26)
    plt.tick_params(labelsize=18)


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
