import joblib


class PipelineHandler:
    def __init__(self, pipeline_path):
        self.pipeline = joblib.load(pipeline_path)

    def __call__(self, flask_request):
        payload = flask_request.json
        pred = self.pipeline.predict(payload['data'])
        # do post-processing stuff
        return pred.tolist()
