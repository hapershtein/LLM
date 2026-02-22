"""Ollama API client with streaming support."""

import json
from typing import Iterator

import httpx

DEFAULT_BASE_URL = "http://localhost:11434"


class OllamaClient:
    def __init__(self, base_url: str = DEFAULT_BASE_URL):
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(timeout=None)

    def list_models(self) -> list[str]:
        try:
            resp = self._client.get(f"{self.base_url}/api/tags")
            resp.raise_for_status()
            return [m["name"] for m in resp.json().get("models", [])]
        except Exception as e:
            raise ConnectionError(f"Cannot reach Ollama at {self.base_url}: {e}") from e

    def chat(
        self,
        model: str,
        messages: list[dict],
        tools: list[dict] | None = None,
        stream: bool = True,
    ) -> Iterator[dict]:
        """
        Yields streaming chunks. Each chunk is the parsed JSON object from
        Ollama's /api/chat streaming response.

        When stream=False (tool-call mode), yields a single dict.
        """
        payload: dict = {
            "model": model,
            "messages": messages,
            "stream": stream,
        }
        if tools:
            payload["tools"] = tools

        with self._client.stream(
            "POST",
            f"{self.base_url}/api/chat",
            json=payload,
            timeout=None,
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                line = line.strip()
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    continue
                yield chunk
                if chunk.get("done"):
                    break

    def close(self):
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
