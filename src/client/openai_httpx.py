import httpx
from typing import Optional

from openai import OpenAI

def build_httpx_client() -> httpx.Client:
    """
    Builds a synchronous httpx client.
    """
    return httpx.Client(
        timeout=60.0,
        follow_redirects=True,
    )

def build_async_httpx_client() -> httpx.AsyncClient:
    """
    Builds an asynchronous httpx client.
    """
    return httpx.AsyncClient(
        timeout=60.0,
        follow_redirects=True,
    )

def run_chat(prompt: str, model: str = "gpt-3.5-turbo", temperature: float = 0.7):
    """
    Simple wrapper for OpenAI chat completions.
    """
    from config import get_openai_config
    config = get_openai_config()
    client = OpenAI(
        api_key=config["api_key"],
        base_url=config["base_url"],
        http_client=build_httpx_client()
    )
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature
    )
    return response.model_dump()
