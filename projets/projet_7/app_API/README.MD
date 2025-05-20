### Commandes installation

# Lancement service Docker
sudo systemctl start docker

# Activation service au boot
sudo systemctl enable docker

# Ajouter utilisateur au groupe decker
sudo usermod -aG docker $USER

# Utilisation de pyenv virtualenvwrapper pour l'environnement en local
pyenv virtualenvwrapper
mkvirtualenv fastapi-env

# Exécuter ligne suivante dans l'environnement
pip install fastapi uvicorn pandas scikit-learn pyarrow shap pytest lightgbm

# Revenir dans le shell
deactivate
#workon fastapi-env

# Construire image docker
docker buildx build -t fastapi-app --load .

# Exécuter conteneur docker
docker run -p 8000:8000 fastapi-app



### Autres

# Après changements, re-build et exécuter si besoin
docker build -t fastapi-app .
docker run -p 8000:8000 fastapi-app

# Si probleme daemon port utilisé : 
sudo systemctl restart docker




### Tests

# Exécution des tests
pyenv virtualenvwrapper
workon fastapi-env
pytest -v tests/

# Exécuter un test spécifique
pytest tests/test_main.py::test_get_data_parquet

# Afficher moins d'infos
pytest tests/test_main.py
