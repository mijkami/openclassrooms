# Nom de l'image Docker
IMAGE_NAME = fastapi-app

# Port sur lequel l'API sera accessible
API_PORT = 8000



# Commande pour construire l'image Docker
build:
	docker build -t $(IMAGE_NAME) .

# Commande pour exécuter le conteneur Docker
run:
	docker run -p $(API_PORT):8000 $(IMAGE_NAME)

# Commande pour arrêter et supprimer le conteneur Docker
stop:
	docker stop $(IMAGE_NAME) || true
	docker rm $(IMAGE_NAME) || true

# Commande pour reconstruire et relancer le conteneur Docker
rebuild:
	make stop
	make build
	make run

# Commande pour nettoyer les images et conteneurs Docker
clean:
	docker rmi $(IMAGE_NAME) || true
