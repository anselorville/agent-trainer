import os

from dotenv import load_dotenv


load_dotenv()


def get_openai_config():
    return {
        "api_key": os.getenv("LLM_API_KEY"),
        "base_url": os.getenv("LLM_BASE_URL"),
        "model_name": os.getenv("LLM_MODEL_NAME", "deepseek-v3.2"),
    }
