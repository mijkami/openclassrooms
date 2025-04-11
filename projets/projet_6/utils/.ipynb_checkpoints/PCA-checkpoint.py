import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
from sklearn.impute import SimpleImputer
#%matplotlib inline

import warnings
warnings.filterwarnings('ignore')
pd.options.mode.chained_assignment = None  # default='warn'

def clean_data(data, select_X=None, impute=False, std=False): 
    """Returns dataframe with selected, imputed 
       and standardized features
    
    Input
          data: dataframe
          select_X: list of feature names to be selected (string)
          impute: If True impute np.nan with mean
          std: If True standardize data
          
    Return
        dataframe: data with selected, imputed 
                   and standardized features    
    """
    
    # (i.) select features
    if select_X is not None:
        data = data.filter(select_X, axis='columns')
        print("\t>>> Selected features: {}".format(select_X))
    else:
        # store column names
        select_X = list(data.columns)
    
    # (ii.) impute with mean 
    if impute:
        imp = SimpleImputer()
        data = imp.fit_transform(data)
        print("\t>>> Imputed missings")
    
    # (iii.) standardize 
    if std:
        std_scaler = StandardScaler()
        data = std_scaler.fit_transform(data)
        print("\t>>> Standardized data : StandardScaler")
    
    return pd.DataFrame(data, columns=select_X)

def myPCA(df, clusters=None, mle_use=True, n_compos=0.95):
    # https://github.com/mazieres/analysis/blob/master/analysis.py#L19-34
    # Normalize data
    df_norm = (df - df.mean()) / df.std()
    # PCA
    if mle_use:
        pca = PCA(n_components='mle')
    else:
        pca = PCA(n_components=n_compos)
        
    pca_res = pca.fit_transform(df_norm.values)
    # Ebouli
    ebouli = pd.Series(pca.explained_variance_ratio_)
    ebouli.plot(kind='bar', title="Ebouli des valeurs propres")
    plt.show()
    # Circle of correlations http://stackoverflow.com/a/22996786/1565438
    coef = np.transpose(pca.components_)
    cols = ['PC-'+str(x) for x in range(len(ebouli))]
    pc_infos = pd.DataFrame(coef, columns=cols, index=df_norm.columns)
    circleOfCorrelations(pc_infos, ebouli)
    plt.show()
    # Plot PCA
    dat = pd.DataFrame(pca_res, columns=cols)
    if isinstance(clusters, np.ndarray):
        for clust in set(clusters):
            colors = list("bgrcmyk")
            plt.scatter(dat["PC-0"][clusters==clust],dat["PC-1"][clusters==clust],c=colors[clust])
    else:
        plt.scatter(dat["PC-0"],dat["PC-1"])
    plt.xlabel("PC-0 (%s%%)" % str(ebouli[0])[:4].lstrip("0."))
    plt.ylabel("PC-1 (%s%%)" % str(ebouli[1])[:4].lstrip("0."))
    plt.title("PCA")
    plt.show()

    # check factor loading matrix
    df_c = pd.DataFrame(pca.components_, columns=df.columns).T

    # matrix adjust y-axis size dynamically
    size_yaxis = round(df.shape[1] * 0.5)
    fig, ax = plt.subplots(figsize=(8,size_yaxis))# plot the first top_pc components
    top_pc = 3
    sns.heatmap(df_c.iloc[:,:top_pc], annot=True, cmap="YlGnBu", ax=ax)
    plt.show()

    # basic info
    n_components = len(pca.explained_variance_ratio_)
    explained_variance = pca.explained_variance_ratio_
    cum_explained_variance = np.cumsum(explained_variance)
    idx = np.arange(n_components)+1
    df_explained_variance = pd.DataFrame([explained_variance, cum_explained_variance], 
                                        index=['explained variance', 'cumulative'], 
                                        columns=idx).T

    mean_explained_variance = df_explained_variance.iloc[:,0].mean() # calculate mean explained variance# 
    
    # Print explained variance as plain text
    print('Résumé PCA')
    print('='*40)
    print("Total: {} composants".format(n_components))
    print('-'*40)
    print('Moyenne de la variance expliquée :', round(mean_explained_variance,3))
    print('-'*40)
    print(df_explained_variance.head(20))
    print('-'*40)


    return pc_infos.style.background_gradient(cmap='YlGnBu')



def circleOfCorrelations(pc_infos, ebouli):
    plt.figure(figsize = (10, 10))
    plt.Circle((0,0),radius=10, color='g', fill=False)
    circle1=plt.Circle((0,0),radius=1, color='g', fill=False)
    fig = plt.gcf()
    fig.gca().add_artist(circle1)
    for idx in range(len(pc_infos["PC-0"])):
        x = pc_infos["PC-0"][idx]
        y = pc_infos["PC-1"][idx]
        plt.plot([0.0,x],[0.0,y],'k-')
        plt.plot(x, y, 'rx')
        plt.annotate(pc_infos.index[idx], xy=(x,y))
    plt.xlabel("PC-0 (%s%%)" % str(ebouli[0])[:4].lstrip("0."), fontsize=15, fontweight='bold')
    plt.ylabel("PC-1 (%s%%)" % str(ebouli[1])[:4].lstrip("0."), fontsize=15, fontweight='bold')
    plt.xlim((-1,1))
    plt.ylim((-1,1))
    plt.title("Circle of Correlations", fontsize=20, fontweight='bold')

def CHA(df, num_cols, label_size=15, rotation=90):
    corr = df[num_cols].dropna().corr()

    plt.figure(figsize=(12,5))
    dissimilarity = 1 - abs(corr)
    Z = linkage(squareform(dissimilarity), 'complete')

    dendrogram(Z, 
            labels=num_cols, 
            orientation='top', 
            leaf_rotation=rotation
            )
    plt.suptitle('CHA : Classification Hiérarchique Ascendante', fontsize=26)
    plt.tick_params(labelsize=label_size)