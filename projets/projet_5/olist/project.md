dbscan ne fonctionne pas => explique pourquoi il n'est pas adapté

exercice de merge de données


rfm


recency frequency monetary value

97% n'ont commandé qu'une seule fois



avant de commencer le clustering faire quelques analyses de features 
=> RFM marketing




monetary value : moyenne ou somme +/- équivalent


rfm => ajouter le review score



test dbscan kmeans rh (clustering)  sur R F M uniquement

=> ensuite lorsque l'on a déterminer meilleur algo (kmeans ?) => appliquer RFM sur d'autre features (ex : review score)


pour déterminer le nb optimal de clusters pour chaque méthode, diff méthodes : coût, silhouette... à comparer




autre objectif : analyser le clustering

ex : kmeans via rfm et review score => analyser les résultats

=> faire graphiques de type 



toute dernière étape, contrat de maintenance

=> score ari

petite simulation ex sur data de 2017, faire clustering
sur données de 2018, ajouter clients 15aine par 15aine ou mois par mois... et à chaque fois calculer l'ARI et plot sur un graphique et un ari inferieur à 0.8 => signe de non-stabilité de clustering