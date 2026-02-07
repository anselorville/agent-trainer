import httpx
from typing import Optional

from openai import OpenAI

def build_httpx_client() -> httpx.Client:
    """
    Builds a synchronous httpx client.
    """
    return httpx.Client(
        timeout=300.0,
        follow_redirects=True,
    )

def build_async_httpx_client() -> httpx.AsyncClient:
    """
    Builds an asynchronous httpx client.
    """
    return httpx.AsyncClient(
        timeout=300.0,
        follow_redirects=True,
    )

def run_chat(prompt: str, model: str = None, temperature: float = 0.7):
    """
    Simple wrapper for OpenAI chat completions.
    """
    from settings import BASE_CONFIG
    client = OpenAI(
        api_key=BASE_CONFIG.api_key,
        base_url=BASE_CONFIG.base_url,
        http_client=build_httpx_client()
    )
    response = client.chat.completions.create(
        model=model or BASE_CONFIG.model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature
    )
    return response.model_dump()
