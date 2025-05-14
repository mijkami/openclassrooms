# Utiliser une image de base Python
FROM python:3.9

# Définir le répertoire de travail
WORKDIR /app

# Copier les fichiers nécessaires
COPY ./app/requirements.txt /app/requirements.txt
COPY ./app/model.pkl /app/model.pkl
COPY ./app/main.py /app/main.py

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Exposer le port
EXPOSE 8000

# Commande pour exécuter l'application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
