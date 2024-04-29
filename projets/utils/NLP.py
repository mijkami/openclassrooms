import numpy as np
import matplotlib.pyplot as plt

from sklearn import manifold
from sklearn import cluster, metrics
from sklearn.decomposition import NMF, LatentDirichletAllocation, MiniBatchNMF

import nltk
nltk.download("stopwords")
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import time

# Tokenizer
def tokenizer_fct(sentence) :
    # print(sentence)
    sentence_clean = sentence.replace('-', ' ').replace('+', ' ').replace('/', ' ').replace('#', ' ')
    word_tokens = word_tokenize(sentence_clean)
    return word_tokens

# Stop words
stop_w = list(set(stopwords.words('english'))) + ['[', ']', ',', '.', ':', '?', '(', ')']

def stop_word_filter_fct(list_words) :
    filtered_w = [w for w in list_words if not w in stop_w]
    filtered_w2 = [w for w in filtered_w if len(w) > 2]
    return filtered_w2

# lower case et alpha
def lower_start_fct(list_words) :
    lw = [w.lower() for w in list_words if (not w.startswith("@")) 
    #                                   and (not w.startswith("#"))
                                       and (not w.startswith("http"))]
    return lw

# Lemmatizer (base d'un mot)
def lemma_fct(list_words) :
    lemmatizer = WordNetLemmatizer()
    lem_w = [lemmatizer.lemmatize(w) for w in list_words]
    return lem_w

# Fonction de préparation du texte pour le bag of words (Countvectorizer et Tf_idf, Word2Vec)
def transform_bow_fct(desc_text) :
    word_tokens = tokenizer_fct(desc_text)
    sw = stop_word_filter_fct(word_tokens)
    lw = lower_start_fct(sw)
    # lem_w = lemma_fct(lw)    
    transf_desc_text = ' '.join(lw)
    return transf_desc_text

# Fonction de préparation du texte pour le bag of words avec lemmatization
def transform_bow_lem_fct(desc_text) :
    word_tokens = tokenizer_fct(desc_text)
    sw = stop_word_filter_fct(word_tokens)
    lw = lower_start_fct(sw)
    lem_w = lemma_fct(lw)    
    transf_desc_text = ' '.join(lem_w)
    return transf_desc_text

# Fonction de préparation du texte pour le Deep learning (USE et BERT)
def transform_dl_fct(desc_text) :
    word_tokens = tokenizer_fct(desc_text)
#    sw = stop_word_filter_fct(word_tokens)
    lw = lower_start_fct(word_tokens)
    # lem_w = lemma_fct(lw)    
    transf_desc_text = ' '.join(lw)
    return transf_desc_text



# Calcul Tsne, détermination des clusters et calcul ARI entre vrais catégorie et n° de clusters
def ARI_fct(features, labels, y_cat_num, perplexity_val=30, verbose=True) :
    time1 = time.time()
    num_labels=len(labels)
    tsne = manifold.TSNE(n_components=2, perplexity=perplexity_val, n_iter=2000, 
                                 init='random', learning_rate=200, random_state=42)
    X_tsne = tsne.fit_transform(features)
    
    # Détermination des clusters à partir des données après Tsne 
    cls = cluster.KMeans(n_clusters=num_labels, n_init=100, random_state=42)
    cls.fit(X_tsne)
    ARI = np.round(metrics.adjusted_rand_score(y_cat_num, cls.labels_),4)
    time2 = np.round(time.time() - time1,0)
    if verbose :
        print("ARI : ", ARI, "time : ", time2)
    
    return ARI, X_tsne, cls.labels_


# visualisation du Tsne selon les vraies catégories et selon les clusters
def TSNE_visu_fct(X_tsne, y_cat_num, labels, lab_cat, ARI, verbose=True, length=15, height=6) :
    fig = plt.figure(figsize=(length,height))
    
    ax = fig.add_subplot(121)
    scatter = ax.scatter(X_tsne[:,0],X_tsne[:,1], c=y_cat_num, cmap='Set1')
    ax.legend(handles=scatter.legend_elements()[0], labels=lab_cat, loc="best", title="Categorie")
    plt.title('Représentation des phrases par catégories réelles')
    
    ax = fig.add_subplot(122)
    scatter = ax.scatter(X_tsne[:,0],X_tsne[:,1], c=labels, cmap='Set1')
    ax.legend(handles=scatter.legend_elements()[0], labels=set(labels), loc="best", title="Clusters")
    plt.title('Représentation des phrases par clusters')
    
    plt.show()
    if verbose:
        print("ARI : ", ARI)

# will be used in next fct for extracting most imported words per topic
# https://scikit-learn.org/stable/auto_examples/applications/plot_topics_extraction_with_nmf_lda.html#sphx-glr-auto-examples-applications-plot-topics-extraction-with-nmf-lda-py
def plot_top_words(model, feature_names, n_top_words, title):
    fig, axes = plt.subplots(2, 4, figsize=(30, 15), sharex=True)
    axes = axes.flatten()
    for topic_idx, topic in enumerate(model.components_):
        top_features_ind = topic.argsort()[-n_top_words:]
        top_features = feature_names[top_features_ind]
        weights = topic[top_features_ind]

        ax = axes[topic_idx]
        ax.barh(top_features, weights, height=0.7)
        ax.set_title(f"Topic {topic_idx +1}", fontdict={"fontsize": 30})
        ax.tick_params(axis="both", which="major", labelsize=20)
        for i in "top right left".split():
            ax.spines[i].set_visible(False)
        fig.suptitle(title, fontsize=40)

    plt.subplots_adjust(top=0.90, bottom=0.05, wspace=0.90, hspace=0.3)
    plt.show()

def NMF_top_words(model, X, n_top_words, n_components, init="nndsvda"):
    nmf = NMF(
        n_components=n_components,
        random_state=1,
        init=init,
        beta_loss="frobenius",
        alpha_W=0.00005,
        alpha_H=0.00005,
        l1_ratio=1,
    ).fit(X)
    
    tfidf_feature_names = model.get_feature_names_out()
    plot_top_words(
        nmf, tfidf_feature_names, n_top_words, "Topics in NMF model (Frobenius norm)"
    )

def top_words(model, X, n_top_words, n_components, plot_model='NMF', init="nndsvda", batch_size=128):
    if plot_model=='NMF':
        nmf = NMF(
            n_components=n_components,
            random_state=1,
            init=init,
            beta_loss="frobenius",
            alpha_W=0.00005,
            alpha_H=0.00005,
            l1_ratio=1,
        ).fit(X)

        model_feature_names = model.get_feature_names_out()
        plot_top_words(
            nmf, model_feature_names, n_top_words, "Topics in NMF model (Frobenius norm)"
        )
    elif plot_model=='MBNMF':
        mbnmf = MiniBatchNMF(
            n_components=n_components,
            random_state=1,
            batch_size=batch_size,
            init=init,
            beta_loss="kullback-leibler",
            alpha_W=0.00005,
            alpha_H=0.00005,
            l1_ratio=0.5,
        ).fit(X)
        model_feature_names = model.get_feature_names_out()
        plot_top_words(
            mbnmf, model_feature_names, n_top_words, "Topics in MiniBatchNMF model (generalized Kullback-Leibler divergence)"
            )
    elif plot_model=='LDA':
        lda = LatentDirichletAllocation(
            n_components=n_components,
            max_iter=5,
            learning_method="online",
            learning_offset=50.0,
            random_state=0,
        ).fit(X)
        model_feature_names = model.get_feature_names_out()
        plot_top_words(
            lda, model_feature_names, n_top_words, "Topics in LDA model"
        )
    else:
        print('Model not avalaible, please check library / function')