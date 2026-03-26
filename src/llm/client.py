from __future__ import annotations

import os
from openai import OpenAI


def get_client() -> OpenAI:
    return OpenAI(
        base_url=os.getenv("OPENAI_BASE_URL", "http://localhost:8000/v1"),
        api_key=os.getenv("OPENAI_API_KEY", "dummy"),
    )


def get_model() -> str:
    return os.getenv("OPENAI_MODEL", "Qwen/Qwen3-1.7B")