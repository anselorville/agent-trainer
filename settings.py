import os

from dotenv import load_dotenv


load_dotenv()


class ModelConfig:
    def __init__(self, api_key, base_url, model_name):
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model_name


OPTIMIZER_CONFIG = ModelConfig(
    api_key=os.getenv("OPTIMIZER_API_KEY"),
    base_url=os.getenv("OPTIMIZER_BASE_URL"),
    model_name=os.getenv("OPTIMIZER_MODEL_NAME", "glm-4.7"),
)

ROLLOUT_CONFIG = ModelConfig(
    api_key=os.getenv("ROLLOUT_API_KEY"),
    base_url=os.getenv("ROLLOUT_BASE_URL"),
    model_name=os.getenv("ROLLOUT_MODEL_NAME", "glm-4.5-flash"),
)

