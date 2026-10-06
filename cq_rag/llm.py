from __future__ import annotations

import json
import os
import re
from typing import Any, Protocol

from openai import OpenAI


class JsonChatClient(Protocol):
    def complete_json(self, model: str, system: str, prompt: str) -> dict[str, Any]: ...


class OpenAIJsonClient:
    def __init__(self, client: OpenAI) -> None:
        self._client = client

    def complete_json(self, model: str, system: str, prompt: str) -> dict[str, Any]:
        response = self._client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content or "{}"
        value = json.loads(re.sub(r"^```json\s*|\s*```$", "", content.strip(), flags=re.I))
        if not isinstance(value, dict):
            raise ValueError("Chat response must be a JSON object")
        return value

    def embed(self, model: str, text: str, dimensions: int) -> list[float]:
        response = self._client.embeddings.create(
            model=model,
            input=text,
            dimensions=dimensions,
            encoding_format="float",
        )
        if len(response.data) != 1:
            raise RuntimeError("Embedding response must contain exactly one vector")
        return response.data[0].embedding


def create_openai_client() -> OpenAIJsonClient:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set the OPENAI_API_KEY environment variable")
    return OpenAIJsonClient(OpenAI(api_key=api_key, timeout=120.0, max_retries=2))