"""Module providing a FastAPI."""

from fastapi import FastAPI
# from api.exceptions import registry_exceptions
from api.routes import route_registry


def get_api() -> FastAPI:
    app = FastAPI(title="Agricultural Crop Image Classification")
    route_registry(app)
    return app
