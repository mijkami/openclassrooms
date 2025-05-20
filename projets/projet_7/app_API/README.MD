# Commandes installation

# utilisation de pyenv virtualenvwrapper pour l'environnement en local

mkvirtualenv fastapi-env

# exécuter ligne suivante dans l'environnement
pip install fastapi uvicorn pandas scikit-learn pyarrow shap pytest lightgbm

# revenir dans le shell
deactivate
#workon fastapi-env

# construire image docker
docker buildx build -t fastapi-app --load .

# exécuter conteneur docker
docker run -p 8000:8000 fastapi-app
