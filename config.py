import os
from enum import Enum
from pydantic_settings import BaseSettings
from pydantic import BaseModel


class EnvTypes(str, Enum):
    PROD = "prod"
    DEV = "dev"


class AppConfig(BaseModel):
    """Application configurations."""

    pass


class GlobalConfig(BaseSettings):
    """Global configurations."""

    # These variables will be loaded from the .env file. However, if
    # there is a shell environment variable having the same name,
    # that will take precedence.

    APP_CONFIG: AppConfig = AppConfig()

    # Environment specific variables do not need the field class
    ENV_MODE: EnvTypes = EnvTypes.PROD
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "agri_crop.log"
    AWS_ACCESS_KEY_ID: str = ""
    AWS_SECRET_ACCESS_KEY: str = ""
    AWS_REGION: str = "us-east-1"
    S3_BUCKET: str = "asistente-virtual-no-one"

    # API Workers, threads
    N_WORKERS: int = 1

    # Inference params
    INFERENCE_DEVICE: int = 0
    TRACKING_SERVER_HOST: str = "localhost"
    PORT: int = 5000
    EXPERIMENT_NAME: str = "Agricultural-crops-experiments"
    PYTORCH_MODEL_PATH: str = 'models/model.pth'


env = os.getenv("ENV_MODE", "dev")
print(f"Env: {env}")
if env == "dev":
    print("Loading Dev")
    cnf = GlobalConfig(_env_file=".dev.env")
else:
    print("Loading Prod")
    cnf = GlobalConfig(_env_file=".prod.env")
