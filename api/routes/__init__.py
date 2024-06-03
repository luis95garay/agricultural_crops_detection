from fastapi import FastAPI

from . import prediction


def route_registry(app: FastAPI):
    app.include_router(prediction.router)
