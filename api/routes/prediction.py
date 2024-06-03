from io import BytesIO

from fastapi import UploadFile, File, Depends
from fastapi.routing import APIRouter
from PIL import Image
import numpy as np

# from lib.mlflow_prediction import MLFlowPrediction
from lib.pytorch_prediction import PytorchPrediction
from api.responses.response import Responses


router = APIRouter(tags=['prediction'])
# mlflow_model = MLFlowPrediction()
pytorch_model = PytorchPrediction()


# def get_mlflow_model():
#     return mlflow_model

def get_pytorch_model():
    return pytorch_model


@router.get("/")
async def index():
    return {'result': 'exito'}


# @router.post("/mlflow_predict")
# async def mlflow_predict(
#     file: UploadFile = File(...),
#     model: MLFlowPrediction = Depends(get_mlflow_model)
# ):
#     image = Image.open(BytesIO(await file.read()))

#     class_name, score = model.predict(image)
#     data =  {"class_name": class_name, "score": score}
#     return Responses.ok(data)


@router.post("/pytorch_predict")
async def pytorch_predict(
    file: UploadFile = File(...),
    model: PytorchPrediction = Depends(get_pytorch_model)
):
    image = Image.open(BytesIO(await file.read()))

    class_name, score = model.predict(image)
    data =  {"class_name": class_name, "score": score}
    return Responses.ok(data)
