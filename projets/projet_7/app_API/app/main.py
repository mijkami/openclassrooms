from fastapi import FastAPI
import pickle
from pydantic import BaseModel
from typing import List

# Charger le modèle
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Initialiser l'application FastAPI
app = FastAPI()

# Définir le modèle de données pour la requête
class QueryData(BaseModel):
    EXT_SOURCE_2: float
    EXT_SOURCE_3: float
    DAYS_EMPLOYED: float
    INSTAL_DPD_MEAN: float
    CODE_GENDER: float
    PAYMENT_RATE: float
    DAYS_EMPLOYED_PERC: float
    DAYS_BIRTH: float
    PREV_CNT_PAYMENT_MEAN: float
    AMT_GOODS_PRICE: float
    AMT_ANNUITY: float
    ANNUITY_INCOME_PERC: float
    INSTAL_AMT_PAYMENT_SUM: float
    APPROVED_AMT_DOWN_PAYMENT_MAX: float
    BURO_AMT_CREDIT_SUM_DEBT_MEAN: float
    NAME_EDUCATION_TYPE_Highereducation: float
    AMT_CREDIT: float
    ACTIVE_DAYS_CREDIT_MAX: float
    BURO_DAYS_CREDIT_MEAN: float
    PREV_APP_CREDIT_PERC_MEAN: float
    INSTAL_PAYMENT_DIFF_MEAN: float
    POS_MONTHS_BALANCE_SIZE: float
    APPROVED_CNT_PAYMENT_MEAN: float
    DAYS_ID_PUBLISH: float
    APPROVED_AMT_ANNUITY_MEAN: float
    ACTIVE_DAYS_CREDIT_ENDDATE_MAX: float
    BURO_DAYS_CREDIT_MAX: float
    TOTALAREA_MODE: float
    INSTAL_PAYMENT_PERC_MEAN: float
    INSTAL_DAYS_ENTRY_PAYMENT_MEAN: float
    PREV_APP_CREDIT_PERC_MIN: float
    INSTAL_AMT_PAYMENT_MIN: float
    BURO_CREDIT_ACTIVE_Closed_MEAN: float
    DAYS_REGISTRATION: float
    APPROVED_AMT_ANNUITY_MAX: float
    INSTAL_AMT_PAYMENT_MEAN: float
    ACTIVE_DAYS_CREDIT_ENDDATE_MEAN: float

@app.post("/predict")
async def predict(query: QueryData):
    # Convertir les données de la requête en un format adapté au modèle
    input_data = [[
        query.EXT_SOURCE_2, query.EXT_SOURCE_3, query.DAYS_EMPLOYED,
        query.INSTAL_DPD_MEAN, query.CODE_GENDER, query.PAYMENT_RATE,
        query.DAYS_EMPLOYED_PERC, query.DAYS_BIRTH, query.PREV_CNT_PAYMENT_MEAN,
        query.AMT_GOODS_PRICE, query.AMT_ANNUITY, query.ANNUITY_INCOME_PERC,
        query.INSTAL_AMT_PAYMENT_SUM, query.APPROVED_AMT_DOWN_PAYMENT_MAX,
        query.BURO_AMT_CREDIT_SUM_DEBT_MEAN, query.NAME_EDUCATION_TYPE_Highereducation,
        query.AMT_CREDIT, query.ACTIVE_DAYS_CREDIT_MAX, query.BURO_DAYS_CREDIT_MEAN,
        query.PREV_APP_CREDIT_PERC_MEAN, query.INSTAL_PAYMENT_DIFF_MEAN,
        query.POS_MONTHS_BALANCE_SIZE, query.APPROVED_CNT_PAYMENT_MEAN,
        query.DAYS_ID_PUBLISH, query.APPROVED_AMT_ANNUITY_MEAN,
        query.ACTIVE_DAYS_CREDIT_ENDDATE_MAX, query.BURO_DAYS_CREDIT_MAX,
        query.TOTALAREA_MODE, query.INSTAL_PAYMENT_PERC_MEAN,
        query.INSTAL_DAYS_ENTRY_PAYMENT_MEAN, query.PREV_APP_CREDIT_PERC_MIN,
        query.INSTAL_AMT_PAYMENT_MIN, query.BURO_CREDIT_ACTIVE_Closed_MEAN,
        query.DAYS_REGISTRATION, query.APPROVED_AMT_ANNUITY_MAX,
        query.INSTAL_AMT_PAYMENT_MEAN, query.ACTIVE_DAYS_CREDIT_ENDDATE_MEAN
    ]]

    # Faire une prédiction avec le modèle
    prediction = model.predict(input_data)
    return {"prediction": prediction.tolist()}
