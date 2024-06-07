import sys
import os
from PIL import Image
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from lib.pytorch_prediction import PytorchPrediction

pytorch_model = PytorchPrediction()


def test_predict_method_ok():
    path = 'C:/Users/luisg/Documents/projects/cifar-CNN/' + \
        'data/Agricultural-crops-new/test/banana/image (15).jpg'
    image = Image.open(path)

    class_name, score = pytorch_model.predict(image)
    assert class_name == "banana"
    assert score > 0
