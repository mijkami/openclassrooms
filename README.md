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

## pyenv virtualenv usage

# install -name: p5 -version: 3.9.7

$ pyenv install 3.9.7
$ pyenv local 3.9.7
$ pyenv which python
$ python --versions
$ pyenv virtualenvwrapper
$ mkvirtualenv --python /home/mijka/.pyenv/versions/3.9.7/bin/python p5
$ which python
$ ipython kernel install --user --name=p5

# gestion kernel /!\

$ workon p5
$ pip install --upgrade ipykernel
$ python -m ipykernel install --user --name=p5 --display-name="p5"


# other commands

$ deactivate
$ workon p5
$ rmvirtualenv p5


$ pip list
$ pip install -r requirements.txt
$ pip freeze > requirements.txt

# check environnement
## dans jupyter

import sys
print(sys.executable)
print(sys.path)


## dans terminal

$ pip --version
$ !pip --version