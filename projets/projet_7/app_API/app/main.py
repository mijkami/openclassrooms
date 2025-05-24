from fastapi import FastAPI, HTTPException, Response
import os
import pandas as pd
import pickle
import shap
from pydantic import BaseModel
from typing import List
import traceback

# Charger les variables d'environnement
MODEL_PATH = os.getenv('MODEL_PATH', 'app/model.pkl')
DATA_PATH = os.getenv('DATA_PATH', 'data/data.parquet.gzip')
DATA_TEST_PATH = os.getenv('DATA_TEST_PATH', 'data/test_df_cleaned.parquet.gzip')

# Charger le modèle
with open(MODEL_PATH, 'rb') as file:
    model = pickle.load(file)
    
# Charger les données
data = pd.read_parquet(DATA_PATH)
data_test = pd.read_parquet(DATA_TEST_PATH)

# Initialiser l'application FastAPI
app = FastAPI()

# Modèle de données pour la requête
class UserID(BaseModel):
    SK_ID_CURR: int

# Définir le modèle de données pour la requête
class QueryData(BaseModel):
    EXT_SOURCE_2: float
    EXT_SOURCE_3: float
    PAYMENT_RATE: float
    INSTAL_DPD_MEAN: float
    INSTAL_AMT_PAYMENT_SUM: float
    AMT_ANNUITY: float
    DAYS_BIRTH: int
    CODE_GENDER: int
    ANNUITY_INCOME_PERC: float
    APPROVED_AMT_DOWN_PAYMENT_MAX: float
    DAYS_EMPLOYED_PERC: float
    BURO_DAYS_CREDIT_MEAN: float
    PREV_CNT_PAYMENT_MEAN: float
    AMT_GOODS_PRICE: float
    BURO_AMT_CREDIT_SUM_DEBT_MEAN: float
    ACTIVE_DAYS_CREDIT_MAX: float
    ACTIVE_DAYS_CREDIT_ENDDATE_MEAN: float
    PREV_APP_CREDIT_PERC_MEAN: float
    INSTAL_PAYMENT_PERC_MEAN: float
    NAME_EDUCATION_TYPE_Highereducation: int
    DAYS_ID_PUBLISH: int
    APPROVED_CNT_PAYMENT_MEAN: float
    BURO_DAYS_CREDIT_MAX: float
    INSTAL_AMT_PAYMENT_MIN: float
    INSTAL_DAYS_ENTRY_PAYMENT_SUM: float
    POS_MONTHS_BALANCE_SIZE: float
    PREV_NAME_CONTRACT_STATUS_Refused_MEAN: float
    DAYS_REGISTRATION: float
    PREV_APP_CREDIT_PERC_MIN: float
    APPROVED_DAYS_DECISION_MIN: float
    INSTAL_AMT_PAYMENT_MEAN: float
    INSTAL_DBD_SUM: float
    INSTAL_AMT_INSTALMENT_MAX: float
    INCOME_CREDIT_PERC: float
    INSTAL_DAYS_ENTRY_PAYMENT_MEAN: float
    INSTAL_PAYMENT_DIFF_MEAN: float


@app.get("/data")
async def get_data(format: str = 'parquet'):
    if format == 'parquet':
        return Response(content=data.to_parquet(), media_type="application/octet-stream")
    elif format == 'csv':
        return Response(content=data.to_csv(index=False), media_type="text/csv")
    else:
        raise HTTPException(status_code=400, detail="Format not supported")
    
@app.get("/data_test")
async def get_data_test(format: str = 'parquet'):
    if format == 'parquet':
        return Response(content=data_test.to_parquet(), media_type="application/octet-stream")
    elif format == 'csv':
        return Response(content=data_test.to_csv(index=False), media_type="text/csv")
    else:
        raise HTTPException(status_code=400, detail="Format not supported")


@app.post("/user_data")
async def get_user_data(user: UserID):
    user_data = data[data['SK_ID_CURR'] == user.SK_ID_CURR]
    if user_data.empty:
        raise HTTPException(status_code=404, detail="User not found")
    return user_data.to_dict(orient='records')
    
    
@app.post("/shap")
async def get_SHAP(user: UserID):
    try:
        user_data = data[data['SK_ID_CURR'] == user.SK_ID_CURR]
        if user_data.empty:
            raise HTTPException(status_code=404, detail="User not found")

        # Calculer les valeurs SHAP
        explainer = shap.Explainer(model)
        shap_values = explainer(user_data.drop(columns=['TARGET', 'SK_ID_CURR']))
        feature_names = shap_values.feature_names

        # Convertir les valeurs SHAP en une liste de dictionnaires avec les noms des colonnes
        shap_values_list = shap_values.values.tolist()
        shap_values_with_names = [
            {feature_names[i]: value for i, value in enumerate(shap_values_list[0])}
        ]

        return shap_values_with_names
    except Exception as e:
        # Log the full traceback for debugging
        traceback.print_exc()
        return {"error": str(e)}


@app.post("/predict")
async def predict(query: QueryData):
    try:
        # Convertir les données de la requête en un format adapté au modèle
        input_data = [[
            query.EXT_SOURCE_2,
            query.EXT_SOURCE_3,
            query.PAYMENT_RATE,
            query.INSTAL_DPD_MEAN,
            query.INSTAL_AMT_PAYMENT_SUM,
            query.AMT_ANNUITY,
            query.DAYS_BIRTH,
            query.CODE_GENDER,
            query.ANNUITY_INCOME_PERC,
            query.APPROVED_AMT_DOWN_PAYMENT_MAX,
            query.DAYS_EMPLOYED_PERC,
            query.BURO_DAYS_CREDIT_MEAN,
            query.PREV_CNT_PAYMENT_MEAN,
            query.AMT_GOODS_PRICE,
            query.BURO_AMT_CREDIT_SUM_DEBT_MEAN,
            query.ACTIVE_DAYS_CREDIT_MAX,
            query.ACTIVE_DAYS_CREDIT_ENDDATE_MEAN,
            query.PREV_APP_CREDIT_PERC_MEAN,
            query.INSTAL_PAYMENT_PERC_MEAN,
            query.NAME_EDUCATION_TYPE_Highereducation,
            query.DAYS_ID_PUBLISH,
            query.APPROVED_CNT_PAYMENT_MEAN,
            query.BURO_DAYS_CREDIT_MAX,
            query.INSTAL_AMT_PAYMENT_MIN,
            query.INSTAL_DAYS_ENTRY_PAYMENT_SUM,
            query.POS_MONTHS_BALANCE_SIZE,
            query.PREV_NAME_CONTRACT_STATUS_Refused_MEAN,
            query.DAYS_REGISTRATION,
            query.PREV_APP_CREDIT_PERC_MIN,
            query.APPROVED_DAYS_DECISION_MIN,
            query.INSTAL_AMT_PAYMENT_MEAN,
            query.INSTAL_DBD_SUM,
            query.INSTAL_AMT_INSTALMENT_MAX,
            query.INCOME_CREDIT_PERC,
            query.INSTAL_DAYS_ENTRY_PAYMENT_MEAN,
            query.INSTAL_PAYMENT_DIFF_MEAN
        ]]

        # Log the number of features
        print(f"Number of features in input data: {len(input_data[0])}")

        # Faire une prédiction avec le modèle
        prediction = model.predict(input_data)
        probabilities = model.predict_proba(input_data)

        return {
            "prediction": prediction.tolist(),
            "probabilities": probabilities.tolist()
        }
    except Exception as e:
        # Log the full traceback for debugging
        traceback.print_exc()
        return {"error": str(e)}