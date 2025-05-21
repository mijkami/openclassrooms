import streamlit as st
import pandas as pd
import requests
import pyarrow.parquet as pq
import io
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import euclidean
import shap

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

# Fonction pour calculer la distance euclidienne entre deux utilisateurs
def calculate_distance(user1, user2, features):
    return euclidean(user1[features], user2[features])

# Interface Streamlit
st.title("Dashboard Utilisateur")

# Initialiser current_page dans st.session_state s'il n'existe pas
if 'current_page' not in st.session_state:
    st.session_state.current_page = 0

# Chargement des données
data = fetch_data('/data')
if data is not None:
    st.subheader("Données générales :")
    # Réorganiser les colonnes pour afficher SK_ID_CURR et TARGET en premier
    cols = ['SK_ID_CURR', 'TARGET'] + [col for col in data.columns if col not in ['SK_ID_CURR', 'TARGET']]
    data = data[cols]

    # Selectbox pour choisir la quantité de lignes affichées
    page_size_options = [10, 15, 20, 25, 30]
    page_size = st.selectbox("Sélectionnez le nombre de lignes à afficher", page_size_options)

    # Pagination
    total_pages = len(data) // page_size + (1 if len(data) % page_size else 0)
    current_page = st.session_state.current_page

    # Boutons pour naviguer entre les pages
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("Précédent") and current_page > 0:
            st.session_state.current_page -= 1
    with col2:
        st.write(f"Page {current_page + 1} / {total_pages}")
    with col3:
        if st.button("Suivant") and current_page < total_pages - 1:
            st.session_state.current_page += 1

    start_idx = current_page * page_size
    end_idx = start_idx + page_size

    # Utiliser un tableau cliquable pour sélectionner un utilisateur
    styled_data = data.iloc[start_idx:end_idx].style.background_gradient(cmap='coolwarm', subset=pd.IndexSlice[:, data.columns.difference(['SK_ID_CURR'])])
    st.dataframe(styled_data)

    # Convertir SK_ID_CURR en int natif
    data['SK_ID_CURR'] = data['SK_ID_CURR'].astype(int)

    # Sélection d'un utilisateur
    st.subheader("Données spécifiques :")
    user_id = st.selectbox("Sélectionnez un utilisateur", data['SK_ID_CURR'])

    # Affichage des données de l'utilisateur sélectionné
    user_data = fetch_data('/user_data', params={'SK_ID_CURR': user_id})
    if user_data is not None:
        # Convertir user_data en DataFrame si ce n'est pas déjà le cas
        if isinstance(user_data, list):
            user_data = pd.DataFrame(user_data)

        # Réorganiser les colonnes pour afficher SK_ID_CURR et TARGET en premier
        user_data_cols = ['SK_ID_CURR', 'TARGET'] + [col for col in user_data.columns if col not in ['SK_ID_CURR', 'TARGET']]
        user_data = user_data[user_data_cols]

        st.write(f"Données de l'utilisateur {user_id} :")
        styled_user_data = user_data.style.background_gradient(cmap='coolwarm', subset=pd.IndexSlice[:, user_data.columns.difference(['SK_ID_CURR'])])
        st.dataframe(styled_user_data)

        # Prédiction
        if st.button("Lancer la prédiction"):
            user_data_df = pd.DataFrame(user_data)
            prediction_data = user_data_df.drop(columns=['SK_ID_CURR', 'TARGET']).to_dict(orient='records')[0]
            prediction = predict(prediction_data)
            clean_prediction = int(prediction['prediction'][0])
            if clean_prediction is not None:
                st.write(f"Prédiction pour l'utilisateur {user_id} : {clean_prediction}")

        # Comparaison avec des utilisateurs similaires
        features = [col for col in data.columns if col not in ['SK_ID_CURR', 'TARGET']]
        user_features = user_data[features].iloc[0]

        # Filtrer les utilisateurs ayant des valeurs proches pour EXT_SOURCE_2, EXT_SOURCE_3, PAYMENT_RATE, et INSTAL_DPD_MEAN
        ext_source_2 = user_features['EXT_SOURCE_2']
        ext_source_3 = user_features['EXT_SOURCE_3']
        payment_rate = user_features['PAYMENT_RATE']

        # Définir une marge pour la similarité
        margin = 0.05
        similar_users = data[
            (data['TARGET'] == 1) &
            (abs(data['EXT_SOURCE_2'] - ext_source_2) <= margin) &
            (abs(data['EXT_SOURCE_3'] - ext_source_3) <= margin) &
            (abs(data['PAYMENT_RATE'] - payment_rate) <= margin) 
        ]

        # Échantillonnage d' utilisateurs
        sample_size = 400
        if len(similar_users) > sample_size:
            similar_users = similar_users.sample(sample_size)

        similar_users['distance'] = similar_users.apply(lambda row: calculate_distance(user_features, row[features], features), axis=1)
        similar_users = similar_users.nsmallest(5, 'distance')

        combined_data = pd.concat([user_data, similar_users], ignore_index=True)
        combined_data_cols = ['SK_ID_CURR', 'TARGET'] + [col for col in combined_data.columns if col not in ['SK_ID_CURR', 'TARGET', 'distance']]
        combined_data = combined_data[combined_data_cols]
        st.write("Données de l'utilisateur sélectionné et d'utilisateurs similaires avec TARGET = 1 :")
        styled_combined_data = combined_data.style.background_gradient(cmap='coolwarm', subset=pd.IndexSlice[:, combined_data.columns.difference(['SK_ID_CURR'])])
        st.dataframe(styled_combined_data)

        # Affichage des données SHAP
        shap_data = fetch_data('/shap', params={'SK_ID_CURR': user_id})
        if shap_data is not None:
            st.write(f"Données SHAP pour l'utilisateur {user_id} :")

            # Extraire les valeurs SHAP et les noms des caractéristiques
            shap_values = np.array([list(shap_data[0].values())])
            feature_names = list(shap_data[0].keys())
            expected_values = np.zeros(len(feature_names))  # Vous pouvez remplacer cela par les valeurs de base réelles si disponibles

            # Créer un objet Explanation de SHAP
            expl = shap.Explanation(values=shap_values,
                                    base_values=expected_values,
                                    feature_names=feature_names)

            # Visualiser les valeurs SHAP
            fig, ax = plt.subplots()
            shap.plots.bar(expl, show=False)
            st.pyplot(fig)
            
        