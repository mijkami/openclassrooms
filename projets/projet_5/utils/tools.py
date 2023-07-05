import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#%matplotlib inline

import warnings
warnings.filterwarnings('ignore')
pd.options.mode.chained_assignment = None  # default='warn'

def UVA_numeric2(data, i):
    '''
    Univariate_Analysis_numeric
    Takes a group of variables (INTEGER and FLOAT) and plot/print 
        all the descriptives and properties along with KDE.
    Runs a loop: calculate all the descriptives of i(th) variable 
        and plot/print it.
    '''
    # calculating descriptives of variable
    mini = data[i].min()
    maxi = data[i].max()
    ran = data[i].max()-data[i].min()
    mean = data[i].mean()
    median = data[i].median()
    st_dev = data[i].std()
    skew = data[i].skew()
    kurt = data[i].kurtosis()

    # calculating points of standard deviation
    points = mean-st_dev, mean+st_dev

    #Plotting the variable with every information
    sns.kdeplot(data[i], shade=True)
    sns.lineplot(points, [0,0], color = 'black', label = "std_dev")
    sns.scatterplot([mini,maxi], [0,0], color = 'orange', label = "min/max")
    sns.scatterplot([mean], [0], color = 'red', label = "mean")
    sns.scatterplot([median], [0], color = 'blue', label = "median")
    plt.xlabel('{}'.format(i), fontsize = 20)
    plt.ylabel('density')
    plt.title('std_dev = {}; kurtosis = {};\nskew = {}; range = {}\nmean = {}; median = {}'\
        .format((round(points[0],2),round(points[1],2)),
                round(kurt,2),
                round(skew,2),
                (round(mini,2),round(maxi,2),round(ran,2)),
                round(mean,2),
                round(median,2)))


def UVA_num_plus(df, indicators_list):
    for i in indicators_list:
        plt.figure(figsize = (15, 20))
        df_indic = df[i]

        plt.subplot(5,2,1)
        sns.boxplot(y=df_indic)
        plt.xlabel(f'{i}', fontsize = 20)

        plt.subplot(5,2,2)
        UVA_numeric2(df, i)


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


def ping():
    """
    You call ping I print pong.
    """
    print('pong')
