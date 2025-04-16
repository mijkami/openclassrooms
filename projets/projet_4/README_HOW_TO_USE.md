# INTRO
Le projet ci-présent peut-être mis en route rapidement en installant un environnement spécifique.

Prérequis :
- gestionnaire d'environnements type pyenv / anaconda
- OU un logiciel permettant de lire/interpréter le notebook enregistré (si juste besoin de lire les résultats).


# A - Lecture / consultation du notebook

- installer vscode : https://code.visualstudio.com/
- ajouter l'extension : https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter
- ouvrir / consulter les fichiers  .ipynb / jupyter notebook avec vscode


# B - Installer l'environnement : Commandes pyenv

## le signe dollar $ indique un début de ligne / nouvelle ligne de code à exécuter

$ pyvenv versions #vérifier les env installés
$ pyenv virtualenv 3.9.7 data_p4
$ pyenv local data_p4
$ pip install -r requirements.txt
$ jupyter notebook



ipython kernel install --user --name=p6_2
