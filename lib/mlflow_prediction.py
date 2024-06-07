import mlflow
from PIL import Image
import numpy as np

from .base_prediction import BasePrediction
from .constants import CLASS_NAMES
from .transformation import transform
from config import cnf


class MLFlowPrediction(BasePrediction):
    def __init__(self):
        super().__init__()
        mlflow.set_tracking_uri(
            f"http://{cnf.TRACKING_SERVER_HOST}:{cnf.PORT}"
        )
        mlflow.set_experiment(cnf.EXPERIMENT_NAME)

        client = mlflow.MlflowClient()
        logged_model_mlf = [
            mv.source for idx, mv in
            enumerate(client.search_model_versions(
                "name='prod.image_classification'"
            )) if idx == 0
        ][0]
        self.logged_model_mlf = mlflow.pyfunc.load_model(logged_model_mlf)

    def predict(self, im: Image):
        transformed_image = transform(im).unsqueeze(0)

        image = transformed_image.numpy()

        output = self.logged_model_mlf.predict(image)

        exp_scores = np.exp(output)
        soft_output = exp_scores / np.sum(exp_scores, keepdims=True)

        result = np.argmax(soft_output, 1)
        score = np.max(soft_output, 1)

        return CLASS_NAMES[result.item()], score.item()
