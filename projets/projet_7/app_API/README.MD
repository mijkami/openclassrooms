### Commandes installation

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



### Autres

# après changements, re-build et exécuter si besoin
docker build -t fastapi-app .
docker run -p 8000:8000 fastapi-app

# si probleme daemon port utilisé : 
sudo systemctl restart docker




### Tests

# Exécution des tests
pytest tests/test_main.py

# Exécuter un test spécifique
pytest tests/test_main.py::test_get_data_parquet

# Afficher plus d'infos
pytest -v tests/
