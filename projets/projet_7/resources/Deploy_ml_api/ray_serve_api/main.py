from ray import serve
from backend import PipelineHandler


def main():
    client = serve.start()
    client.create_backend("house-price:v1", PipelineHandler, 'model.pkl')
    client.create_endpoint(
        "house_pricing", backend="house-price:v1", route="/regressor", methods=['POST'])

    # Listez l'ensemble des backends actifs depuis la console Python.
    client.list_backends()

    # Listez l'ensemble des endpoints actifs depuis la console Python.
    client.list_endpoints()

    # pour mettre à jour le traffic
    client.create_backend("house-price:v2", PipelineHandler, 'model.pkl')
    client.set_traffic("house_pricing", {
                       "house-price:v1": 0.25, "house-price:v2": 0.75})

    # augmentation du nombre de répliques
    config = {"num_replicas": 5}
    client.update_backend_config("house-price:v2", config)


if __name__ == '__main__':
    main()
