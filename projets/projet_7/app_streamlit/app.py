import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import shap
from io import BytesIO
import numpy as np

# Configuration de l'API
API_BASE_URL = "http://127.0.0.1:8000/"

# Fonction pour récupérer les données de l'API
def fetch_data(endpoint):
    response = requests.get(f"{API_BASE_URL}{endpoint}")
    if response.status_code == 200:
        return response.content
    else:
        st.error("Erreur lors de la récupération des données")
        return None

# Fonction pour récupérer les données SHAP d'un utilisateur
def fetch_shap_data(sk_id_curr):
    try:
        # Envoyer une requête POST avec le SK_ID_CURR
        response = requests.post(f"{API_BASE_URL}shap", json={"SK_ID_CURR": int(sk_id_curr)})
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Erreur lors de la récupération des données SHAP: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"Exception lors de la récupération des données SHAP: {e}")
        return None

# Fonction pour faire une prédiction
def predict(data):
    response = requests.post(f"{API_BASE_URL}predict", json=data)
    if response.status_code == 200:
        return response.json()
    else:
        st.error("Erreur lors de la prédiction")
        return None

# Chargement des données
data = fetch_data("data")
if data is not None:
    df = pd.read_parquet(BytesIO(data))

# Interface utilisateur
st.title("Dashboard de Prédiction")

# Sélection de l'utilisateur
sk_id_curr = st.selectbox("Sélectionnez un ID client", df["SK_ID_CURR"].unique())

# Chargement des données de l'utilisateur sélectionné
user_data = df[df["SK_ID_CURR"] == sk_id_curr].iloc[0]

# Checkbox pour afficher ou cacher les champs modifiables
show_fields = st.checkbox("Afficher les champs modifiables")

if show_fields:
    # Affichage des données de l'utilisateur
    st.subheader("Données de l'utilisateur")

    # Utilisation de colonnes pour afficher les champs de manière compacte
    cols = st.columns(3)  # Crée 3 colonnes
    for i, col in enumerate(df.columns):
        if col != "SK_ID_CURR" and col != "TARGET":
            # Utilise modulo pour répartir les champs dans les colonnes
            user_data[col] = cols[i % 3].text_input(col, user_data[col], key=f"{col}_{i}")

# Bouton pour lancer la prédiction
if st.button("Lancer la prédiction"):
    # Mettre à jour user_data avec les nouvelles valeurs saisies par l'utilisateur
    updated_user_data = user_data.copy()
    if show_fields:
        for i, col in enumerate(df.columns):
            if col != "SK_ID_CURR" and col != "TARGET":
                # Utiliser une clé unique pour chaque appel de st.text_input
                updated_user_data[col] = cols[i % 3].text_input(col, user_data[col], key=f"{col}_{i}_pred")

    prediction = predict(updated_user_data.to_dict())
    st.write(f"Prédiction: {prediction}")

# Affichage des données SHAP
st.subheader("Données SHAP")
shap_data = fetch_shap_data(sk_id_curr)
if shap_data is not None:
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

    # Ajouter un toggle pour afficher ou cacher l'intégralité du SHAP
    show_full_shap = st.checkbox("Afficher l'intégralité du SHAP")

    if show_full_shap:
        # Assurez-vous que les dimensions des matrices shap_values et user_data correspondent
        user_data_df = user_data.to_frame().T[feature_names]
        plt.figure()
        shap.summary_plot(shap_values, user_data_df, plot_type="bar")
        st.pyplot(plt.gcf())
        plt.close()

    # Trier les variables par importance SHAP
    shap_importance = pd.Series(shap_values[0], index=feature_names).abs().sort_values(ascending=False)
    top_variables = shap_importance.index.tolist()

# Calculer la médiane et la distance à la médiane
median = df.median()
distance_from_median = np.abs(user_data - median)
far_from_median = distance_from_median.sort_values(ascending=False).index.tolist()

# Affichage des distributions des variables
st.subheader("Positionnement du client par variable")

# Ajouter un bouton pour basculer entre les modes
simple_mode = st.checkbox("Mode manuel", value=True)

if simple_mode:
    # Menu de sélection pour choisir les variables à afficher
    selected_vars = st.multiselect("Sélectionnez les variables à afficher",
                                  options=top_variables,
                                  default=top_variables[:3])
else:
    # Ajouter un slider pour choisir le nombre de variables à afficher
    max_n_variables = len(top_variables) if 'top_variables' in locals() else 3
    n_variables = st.slider("Nombre de variables à afficher", min_value=3, max_value=max_n_variables, value=3)

    # Utiliser des colonnes pour afficher les boutons horizontalement
    button_cols = st.columns(2)
    with button_cols[0]:
        if st.button("Afficher les variables éloignées de la médiane"):
            st.session_state.display_mode = "far_from_median"

    with button_cols[1]:
        if st.button("Afficher le top SHAP"):
            st.session_state.display_mode = "top_shap"

    # Afficher le mode d'affichage actuel
    if st.session_state.display_mode == "far_from_median":
        st.write("Top variables éloignées de la médiane")
        selected_variables = [var for var in far_from_median[:n_variables] if var != "SK_ID_CURR"]
    else:
        st.write("Top variables du SHAP")
        selected_variables = [var for var in top_variables[:n_variables] if var != "SK_ID_CURR"]

# Utiliser deux colonnes pour afficher les distributions
dist_cols = st.columns(2)
if simple_mode:
    selected_variables = selected_vars
else:
    selected_variables = selected_variables

for i, var in enumerate(selected_variables):
    fig, ax = plt.subplots(figsize=(6, 4))  # Créer une nouvelle figure et un axe
    df[var].hist(ax=ax, bins=30)
    ax.axvline(user_data[var], color='red', linestyle='dashed', linewidth=3)
    ax.set_title(f"Distribution de {var}")  # Ajouter un titre à chaque graphique
    dist_cols[i % 2].pyplot(fig)
    plt.close(fig)  # Fermer la figure après utilisation
