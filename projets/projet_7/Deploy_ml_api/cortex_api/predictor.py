import os
import joblib


class PythonPredictor:
    def __init__(self, config):
        self.model = joblib.load('pipeline_housing.joblib')

    def predict(self, payload):
        # do pre-processing stuff
        pred = self.model.predict(payload['data'])
        # do post-processing stuff
        return pred.tolist()
