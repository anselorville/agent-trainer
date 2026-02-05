import os

from dotenv import load_dotenv


load_dotenv()


def get_openai_config():
    return {
        "api_key": os.getenv("OPENAI_API_KEY"),
        "base_url": os.getenv("OPENAI_BASE_URL"),
        "model_name": os.getenv("OPENAI_MODEL_NAME", "gpt-5-mini"),
    }
