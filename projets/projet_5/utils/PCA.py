import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#%matplotlib inline

import warnings
warnings.filterwarnings('ignore')
pd.options.mode.chained_assignment = None  # default='warn'



def myPCA(df, clusters=None):
    # https://github.com/mazieres/analysis/blob/master/analysis.py#L19-34
    # Normalize data
    df_norm = (df - df.mean()) / df.std()
    # PCA
    pca = PCA(n_components='mle')
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
    return pc_infos, ebouli



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
