import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
#%matplotlib inline

import statsmodels.formula.api as smf
import statsmodels.api as sm

import warnings
warnings.filterwarnings('ignore')
pd.options.mode.chained_assignment = None  # default='warn'

def displayColumnDistribution(df):
  i=1
  for col in df.columns:
    plt.subplot(4, 4, i)
    plt.title('Column ' + str(col) + ', Var=' + str(np.var(df[col])))
    x = np.arange(df.shape[0])
    plt.bar(x, df[col])
    i+=1

def boxplot_nums(df, cols_lists, fig_w=11.7, fig_h=8.27, log=False):
    sns.set(rc={'figure.figsize':(fig_w,fig_h)})
    col_lists_len = len(cols_lists)
    all_lists_len = []
    iter = 0
    for i in cols_lists:
        all_lists_len.append(len(i))
    
    fig, ax  = plt.subplots(1,col_lists_len
            #, gridspec_kw={'width_ratios': all_lists_len}
            )
    
    for i in cols_lists:
        if log==True:
            sns.boxplot(data=np.log(df[i]), ax=ax[iter]
                        , showmeans=True
                        , meanprops={"marker":"s","markerfacecolor":"white", "markeredgecolor":"green"}
                       ).set(xlabel=i[:11])
        else:
            sns.boxplot(data=df[i], ax=ax[iter]
                        , showmeans=True
                        , meanprops={"marker":"s","markerfacecolor":"white", "markeredgecolor":"green"}
                       ).set(xlabel=i[:11])
        #ax[iter].set_xticklabels(ax[iter].get_xticklabels(),rotation=55)
        ax[iter].tick_params(axis='x', rotation=90)
        iter+=1
    #plt.xticks(rotation=45)
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

def heatmap_num_2(df):
    f, ax = plt.subplots(figsize=(13, 18))
    corr_matrix = df.corr()
    mask = np.zeros_like(corr_matrix, dtype=np.bool_)
    mask[np.triu_indices_from(mask)] = True 
    heatmap = sns.heatmap(corr_matrix,
                        mask = mask,
                        square = True,
                        linewidths = .5,
                        cmap = "coolwarm",
                        cbar_kws = {'shrink': .4,
                                    "ticks" : [-1, -.5, 0, 0.5, 1]},
                        vmin = -1,
                        vmax = 1,
                        annot = True,
                        annot_kws = {"size": 12})

    #add the column names as labels
    ax.set_yticklabels(corr_matrix.columns, rotation = 0)
    ax.set_xticklabels(corr_matrix.columns)
    sns.set_style({'xtick.bottom': True}, {'ytick.left': True})
    plt.show()

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


def dendrogram(df):
    '''
    Plots a dendrogram from correlation matrix.
    '''
    num_cols = df.select_dtypes(include=['int64','float64','Int64', 'int32'])[:].columns.tolist()
    corr = df[num_cols].corr()

    plt.figure(figsize=(12,5))
    dissimilarity = 1 - abs(corr)    
    
    #Z = linkage(squareform(dissimilarity), method='ward')
    Z = linkage(squareform(dissimilarity), 'complete')

    dendrogram(Z, 
            labels=num_cols, 
            orientation='top', 
            leaf_rotation=75
            )
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


def anova_cat(df, var_num, var_cat):
    anova_category = smf.ols(f'{var_num} ~ {var_cat}', data=df).fit()
    fisher_df = sm.stats.anova_lm(anova_category, typ=2)
    print(f"ANOVA {var_num} ~ {var_cat}")
    print('-' * 85)
    print(anova_category.summary().tables[0])
    print("")
    print("Test de Fisher / Tableau d'analyse de la variance : \n")
    print(sm.stats.anova_lm(anova_category, typ=2))
    print('-' * 85)
    print("")
    print('R2: ', round(anova_category.rsquared, 3))
    print(f"p-value: {fisher_df['PR(>F)'].iloc[0]}")


def anova_multiple(df, var_num_list, var_cat):
    dict = {"Anova" :[]
        ,"R2": []
        ,"p-value": []
       }

    for var_num in var_num_list:
        anova_category = smf.ols(f'{var_num} ~ {var_cat}', data=df).fit()
        fisher_df = sm.stats.anova_lm(anova_category, typ=2)
        dict["Anova"].append(f'{var_num} ~ {var_cat}')
        dict["R2"].append(f'{round(anova_category.rsquared, 3)}')
        dict["p-value"].append(f"{fisher_df['PR(>F)'].iloc[0]}")
    
    anova_df = pd.DataFrame.from_dict(dict)
    anova_df = anova_df.astype({'R2':'float', 'p-value':'float'})
    return anova_df.sort_values(by='R2', ascending=False)

def ping():
    """
    You call ping I print pong.
    """
    print('pong')
