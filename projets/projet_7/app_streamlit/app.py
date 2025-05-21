import streamlit as st
import pandas as pd
import requests
import pyarrow.parquet as pq
import io

# Configuration de l'API
API_BASE_URL = "http://127.0.0.1:8000/"

# Fonction pour récupérer les données de l'API
def fetch_data(endpoint, params=None):
    # Utiliser la méthode HTTP correcte pour chaque point de terminaison
    if endpoint == '/data':
        response = requests.get(f"{API_BASE_URL}{endpoint}", params=params)
    elif endpoint == '/user_data':
        response = requests.post(f"{API_BASE_URL}{endpoint}", json=params)  # Utiliser POST pour /user_data
    elif endpoint == '/shap':
        response = requests.post(f"{API_BASE_URL}{endpoint}", json=params)  # Utiliser POST pour /shap
    else:
        response = requests.get(f"{API_BASE_URL}{endpoint}", params=params)

    if response.status_code == 200:
        if endpoint == '/data':
            return pq.read_table(io.BytesIO(response.content)).to_pandas()
        elif endpoint == '/shap':
            return response.json()
        else:
            return response.json()
    else:
        st.error(f"Error fetching data: {response.status_code}")
        return None

# Fonction pour envoyer une prédiction
def predict(data):
    response = requests.post(f"{API_BASE_URL}/predict", json=data)
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Error making prediction: {response.status_code}")
        return None

# Interface Streamlit
st.title("Dashboard Utilisateur")

# Chargement des données
data = fetch_data('/data')
if data is not None:
    # Réorganiser les colonnes pour afficher SK_ID_CURR en premier
    cols = ['SK_ID_CURR'] + [col for col in data.columns if col != 'SK_ID_CURR']
    data = data[cols]

    st.write("Données des utilisateurs :")
    # Utiliser un tableau cliquable pour sélectionner un utilisateur
    st.dataframe(data.head(15))

    # Convertir SK_ID_CURR en int natif
    data['SK_ID_CURR'] = data['SK_ID_CURR'].astype(int)

    # Sélection d'un utilisateur
    user_id = st.selectbox("Sélectionnez un utilisateur", data['SK_ID_CURR'])

    # Affichage des données de l'utilisateur sélectionné
    user_data = fetch_data('/user_data', params={'SK_ID_CURR': user_id})
    if user_data is not None:
        st.write(f"Données de l'utilisateur {user_id} :")
        st.dataframe(user_data)

        # Prédiction
        if st.button("Lancer la prédiction"):
            user_data_df = pd.DataFrame(user_data)
            prediction_data = user_data_df.drop(columns=['SK_ID_CURR', 'TARGET']).to_dict(orient='records')[0]
            prediction = predict(prediction_data)
            clean_prediction = int(prediction['prediction'][0])
            if clean_prediction is not None:
                st.write(f"Prédiction pour l'utilisateur {user_id} : {clean_prediction}")

        # Affichage des données SHAP
        shap_data = fetch_data('/shap', params={'SK_ID_CURR': user_id})
        if shap_data is not None:
            st.write(f"Données SHAP pour l'utilisateur {user_id} :")
            st.line_chart(shap_data[0])

        # Comparaison avec des utilisateurs similaires
        similar_users = data[data['TARGET'] == 1].sample(5)
        st.write("Utilisateurs similaires avec TARGET = 1 :")
        st.dataframe(similar_users)
