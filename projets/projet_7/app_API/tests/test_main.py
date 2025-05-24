from fastapi.testclient import TestClient
from app.main import app
import pandas as pd
import pytest

client = TestClient(app)

VALID_SK_ID_CURR = 442623

valid_query_data = {
    "EXT_SOURCE_2": 0.5,
    "EXT_SOURCE_3": 0.5,
    "PAYMENT_RATE": 0.5,
    "INSTAL_DPD_MEAN": 0.5,
    "INSTAL_AMT_PAYMENT_SUM": 0.5,
    "AMT_ANNUITY": 0.5,
    "DAYS_BIRTH": -10000,
    "CODE_GENDER": 1,
    "ANNUITY_INCOME_PERC": 0.5,
    "APPROVED_AMT_DOWN_PAYMENT_MAX": 0.5,
    "DAYS_EMPLOYED_PERC": 0.5,
    "BURO_DAYS_CREDIT_MEAN": 0.5,
    "PREV_CNT_PAYMENT_MEAN": 0.5,
    "AMT_GOODS_PRICE": 0.5,
    "BURO_AMT_CREDIT_SUM_DEBT_MEAN": 0.5,
    "ACTIVE_DAYS_CREDIT_MAX": 0.5,
    "ACTIVE_DAYS_CREDIT_ENDDATE_MEAN": 0.5,
    "PREV_APP_CREDIT_PERC_MEAN": 0.5,
    "INSTAL_PAYMENT_PERC_MEAN": 0.5,
    "NAME_EDUCATION_TYPE_Highereducation": 1,
    "DAYS_ID_PUBLISH": -1000,
    "APPROVED_CNT_PAYMENT_MEAN": 0.5,
    "BURO_DAYS_CREDIT_MAX": 0.5,
    "INSTAL_AMT_PAYMENT_MIN": 0.5,
    "INSTAL_DAYS_ENTRY_PAYMENT_SUM": 0.5,
    "POS_MONTHS_BALANCE_SIZE": 0.5,
    "PREV_NAME_CONTRACT_STATUS_Refused_MEAN": 0.5,
    "DAYS_REGISTRATION": -1000,
    "PREV_APP_CREDIT_PERC_MIN": 0.5,
    "APPROVED_DAYS_DECISION_MIN": 0.5,
    "INSTAL_AMT_PAYMENT_MEAN": 0.5,
    "INSTAL_DBD_SUM": 0.5,
    "INSTAL_AMT_INSTALMENT_MAX": 0.5,
    "INCOME_CREDIT_PERC": 0.5,
    "INSTAL_DAYS_ENTRY_PAYMENT_MEAN": 0.5,
    "INSTAL_PAYMENT_DIFF_MEAN": 0.5
}


# Données d'entrée invalides pour les tests
invalid_query_data = {
    "EXT_SOURCE_2": "invalid_value",  # Valeur invalide
    "EXT_SOURCE_3": 0.5,
    "PAYMENT_RATE": 0.5,
    "INSTAL_DPD_MEAN": 0.5,
    "INSTAL_AMT_PAYMENT_SUM": 0.5,
    "AMT_ANNUITY": 0.5,
    "DAYS_BIRTH": -10000,
    "CODE_GENDER": 1,
    "ANNUITY_INCOME_PERC": 0.5,
    "APPROVED_AMT_DOWN_PAYMENT_MAX": 0.5,
    "DAYS_EMPLOYED_PERC": 0.5,
    "BURO_DAYS_CREDIT_MEAN": 0.5,
    "PREV_CNT_PAYMENT_MEAN": 0.5,
    "AMT_GOODS_PRICE": 0.5,
    "BURO_AMT_CREDIT_SUM_DEBT_MEAN": 0.5,
    "ACTIVE_DAYS_CREDIT_MAX": 0.5,
    "ACTIVE_DAYS_CREDIT_ENDDATE_MEAN": 0.5,
    "PREV_APP_CREDIT_PERC_MEAN": 0.5,
    "INSTAL_PAYMENT_PERC_MEAN": 0.5,
    "NAME_EDUCATION_TYPE_Highereducation": 1,
    "DAYS_ID_PUBLISH": -1000,
    "APPROVED_CNT_PAYMENT_MEAN": 0.5,
    "BURO_DAYS_CREDIT_MAX": 0.5,
    "INSTAL_AMT_PAYMENT_MIN": 0.5,
    "INSTAL_DAYS_ENTRY_PAYMENT_SUM": 0.5,
    "POS_MONTHS_BALANCE_SIZE": 0.5,
    "PREV_NAME_CONTRACT_STATUS_Refused_MEAN": 0.5,
    "DAYS_REGISTRATION": -1000,
    "PREV_APP_CREDIT_PERC_MIN": 0.5,
    "APPROVED_DAYS_DECISION_MIN": 0.5,
    "INSTAL_AMT_PAYMENT_MEAN": 0.5,
    "INSTAL_DBD_SUM": 0.5,
    "INSTAL_AMT_INSTALMENT_MAX": 0.5,
    "INCOME_CREDIT_PERC": 0.5,
    "INSTAL_DAYS_ENTRY_PAYMENT_MEAN": 0.5,
    "INSTAL_PAYMENT_DIFF_MEAN": 0.5
}


# Test pour l'endpoint /data
def test_get_data_parquet():
    response = client.get("/data?format=parquet")
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/octet-stream"

def test_get_data_csv():
    response = client.get("/data?format=csv")
    assert response.status_code == 200
    assert response.headers["content-type"] == "text/csv; charset=utf-8"

def test_get_data_unsupported_format():
    response = client.get("/data?format=json")
    assert response.status_code == 400
    assert response.json() == {"detail": "Format not supported"}
    
# Test pour l'endpoint /data_test
def test_get_data_test_parquet():
    response = client.get("/data_test?format=parquet")
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/octet-stream"

def test_get_data_test_csv():
    response = client.get("/data_test?format=csv")
    assert response.status_code == 200
    assert response.headers["content-type"] == "text/csv; charset=utf-8"

def test_get_data_test_unsupported_format():
    response = client.get("/data_test?format=json")
    assert response.status_code == 400
    assert response.json() == {"detail": "Format not supported"}

# Test pour l'endpoint /user_data
def test_get_user_data():
    response = client.post("/user_data", json={"SK_ID_CURR": VALID_SK_ID_CURR})
    assert response.status_code == 200
    assert isinstance(response.json(), list)

def test_get_user_data_not_found():
    response = client.post("/user_data", json={"SK_ID_CURR": 99999})
    assert response.status_code == 404
    assert response.json() == {"detail": "User not found"}

# Test pour l'endpoint /shap
def test_get_shap():
    response = client.post("/shap", json={"SK_ID_CURR": VALID_SK_ID_CURR})
    assert response.status_code == 200
    assert isinstance(response.json(), list)

def test_get_shap_not_found():
    response = client.post("/shap", json={"SK_ID_CURR": 99999})
    assert response.status_code == 200
    assert response.json() == {"error": "404: User not found"}
    
def test_get_shap():
    response = client.post("/shap", json={"SK_ID_CURR": VALID_SK_ID_CURR})
    assert response.status_code == 200
    assert isinstance(response.json(), list)
    assert len(response.json()) > 0
    for item in response.json():
        assert isinstance(item, dict)
        for key, value in item.items():
            assert isinstance(value, (int, float))

def test_get_shap_not_found():
    response = client.post("/shap", json={"SK_ID_CURR": 99999})
    assert response.status_code == 200
    assert response.json() == {"error": "404: User not found"}

# Tests pour l'endpoint /shap_by_input 
def test_get_shap_by_input():
    response = client.post("/shap_by_input", json=valid_query_data)
    assert response.status_code == 200
    assert isinstance(response.json(), list)
    assert len(response.json()) > 0
    for item in response.json():
        assert isinstance(item, dict)
        for key, value in item.items():
            assert isinstance(key, str)
            assert isinstance(value, (int, float))

def test_get_shap_by_input_invalid_input():
    response = client.post("/shap_by_input", json=invalid_query_data)
    assert response.status_code == 422  # Code de statut pour les erreurs de validation

def test_get_shap_by_input_response_structure():
    response = client.post("/shap_by_input", json=valid_query_data)
    assert response.status_code == 200
    response_data = response.json()
    assert isinstance(response_data, list)
    assert len(response_data) > 0
    for item in response_data:
        assert isinstance(item, dict)
        for key, value in item.items():
            assert isinstance(key, str)
            assert isinstance(value, (int, float))

def test_get_shap_by_input_performance():
    import time
    start_time = time.time()
    response = client.post("/shap_by_input", json=valid_query_data)
    end_time = time.time()
    assert response.status_code == 200
    assert end_time - start_time < 5.0  # Vérifie que le temps de réponse est inférieur à 5 secondes


# Tests pour l'endpoint /predict
def test_predict():
    response = client.post("/predict", json=valid_query_data)
    assert response.status_code == 200
    response_data = response.json()
    assert "prediction" in response_data
    assert "probabilities" in response_data

def test_predict_invalid_input_data():
    response = client.post("/predict", json=invalid_query_data)
    assert response.status_code == 422  # Code de statut pour les erreurs de validation

def test_predict_response_structure():
    response = client.post("/predict", json=valid_query_data)
    assert response.status_code == 200
    response_data = response.json()

    # Vérifiez que "prediction" est une liste contenant un flottant
    assert isinstance(response_data["prediction"], list)
    assert len(response_data["prediction"]) == 1
    assert isinstance(response_data["prediction"][0], float)

    # Vérifiez que le flottant dans "prediction" est soit 0, soit 1
    assert response_data["prediction"][0] in [0.0, 1.0]

    # Vérifiez que "probabilities" est une liste contenant une liste de flottants
    assert isinstance(response_data["probabilities"], list)
    assert len(response_data["probabilities"]) == 1
    assert isinstance(response_data["probabilities"][0], list)
    for prob in response_data["probabilities"][0]:
        assert isinstance(prob, float)
        assert 0 <= prob <= 1  # Vérifie que les probabilités sont entre 0 et 1


