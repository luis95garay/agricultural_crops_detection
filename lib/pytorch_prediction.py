from PIL import Image
import torch

from .base_prediction import BasePrediction
from .constants import CLASS_NAMES
from .transformation import transform
from config import cnf


class PytorchPrediction(BasePrediction):
    def __init__(self):
        super().__init__()
        
        self.loaded_model = torch.load(cnf.PYTORCH_MODEL_PATH, map_location=torch.device('cuda'))

    def predict(self, im: Image):
        self.loaded_model.eval()
        transformed_image = transform(im).unsqueeze(0)

        prediction = self.loaded_model(transformed_image.cuda()).squeeze(0).softmax(0)

        class_id = prediction.argmax().item()
        score = prediction.max().item()

        return CLASS_NAMES[class_id], score
