"""Unit tests for ollama_client.py."""

import sys
import json
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from ollama_client import OllamaClient


# ── Helpers ────────────────────────────────────────────────────────────────────

def make_httpx_response(json_body: dict = None, status_code: int = 200):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_body or {}
    resp.raise_for_status = MagicMock()
    return resp


def make_stream_response(lines: list[str]):
    """Mock a streaming httpx response that yields the given lines."""
    ctx = MagicMock()
    ctx.__enter__ = MagicMock(return_value=ctx)
    ctx.__exit__ = MagicMock(return_value=False)
    ctx.raise_for_status = MagicMock()
    ctx.iter_lines = MagicMock(return_value=iter(lines))
    return ctx


# ── OllamaClient.__init__ ──────────────────────────────────────────────────────

class TestOllamaClientInit:
    def test_default_base_url(self):
        client = OllamaClient()
        assert client.base_url == "http://localhost:11434"

    def test_custom_base_url(self):
        client = OllamaClient("http://myhost:9999")
        assert client.base_url == "http://myhost:9999"

    def test_trailing_slash_stripped(self):
        client = OllamaClient("http://localhost:11434/")
        assert client.base_url == "http://localhost:11434"

    def test_context_manager(self):
        client = OllamaClient()
        with client as c:
            assert c is client


# ── OllamaClient.list_models ───────────────────────────────────────────────────

class TestListModels:
    def test_returns_model_names(self):
        resp = make_httpx_response({"models": [{"name": "llama3.2"}, {"name": "qwen2.5"}]})
        with patch("ollama_client.httpx.Client") as MockClient:
            MockClient.return_value._client = MagicMock()
            instance = OllamaClient()
            instance._client = MagicMock()
            instance._client.get.return_value = resp

            models = instance.list_models()

        assert models == ["llama3.2", "qwen2.5"]

    def test_empty_models_list(self):
        resp = make_httpx_response({"models": []})
        instance = OllamaClient()
        instance._client = MagicMock()
        instance._client.get.return_value = resp

        assert instance.list_models() == []

    def test_missing_models_key_returns_empty(self):
        resp = make_httpx_response({})
        instance = OllamaClient()
        instance._client = MagicMock()
        instance._client.get.return_value = resp

        assert instance.list_models() == []

    def test_connection_error_raises(self):
        instance = OllamaClient()
        instance._client = MagicMock()
        instance._client.get.side_effect = Exception("connection refused")

        with pytest.raises(ConnectionError) as exc_info:
            instance.list_models()
        assert "Cannot reach Ollama" in str(exc_info.value)

    def test_hits_correct_endpoint(self):
        resp = make_httpx_response({"models": []})
        instance = OllamaClient("http://myhost:1234")
        instance._client = MagicMock()
        instance._client.get.return_value = resp

        instance.list_models()
        instance._client.get.assert_called_once_with("http://myhost:1234/api/tags")


# ── OllamaClient.chat ──────────────────────────────────────────────────────────

class TestChat:
    def _make_chunks(self, payloads: list[dict]) -> list[str]:
        return [json.dumps(p) for p in payloads]

    def test_yields_parsed_chunks(self):
        lines = self._make_chunks([
            {"message": {"role": "assistant", "content": "hello"}, "done": False},
            {"message": {"role": "assistant", "content": ""}, "done": True, "done_reason": "stop"},
        ])
        stream_ctx = make_stream_response(lines)
        instance = OllamaClient()
        instance._client = MagicMock()
        instance._client.stream.return_value = stream_ctx

        chunks = list(instance.chat(model="test", messages=[]))
        assert len(chunks) == 2
        assert chunks[0]["message"]["content"] == "hello"
        assert chunks[1]["done"] is True

    def test_stops_at_done_chunk(self):
        lines = self._make_chunks([
            {"message": {"content": "a"}, "done": False},
            {"message": {"content": "b"}, "done": True},
            {"message": {"content": "c"}, "done": False},  # Should not be yielded
        ])
        stream_ctx = make_stream_response(lines)
        instance = OllamaClient()
        instance._client = MagicMock()
        instance._client.stream.return_value = stream_ctx

        chunks = list(instance.chat(model="test", messages=[]))
        assert len(chunks) == 2
        contents = [c["message"]["content"] for c in chunks]
        assert "c" not in contents

    def test_skips_empty_lines(self):
        lines = ["", "  ", json.dumps({"message": {"content": "x"}, "done": True})]
        stream_ctx = make_stream_response(lines)
        instance = OllamaClient()
        instance._client = MagicMock()
        instance._client.stream.return_value = stream_ctx

        chunks = list(instance.chat(model="test", messages=[]))
        assert len(chunks) == 1

    def test_skips_invalid_json_lines(self):
        lines = [
            "not json at all",
            json.dumps({"message": {"content": "valid"}, "done": True}),
        ]
        stream_ctx = make_stream_response(lines)
        instance = OllamaClient()
        instance._client = MagicMock()
        instance._client.stream.return_value = stream_ctx

        chunks = list(instance.chat(model="test", messages=[]))
        assert len(chunks) == 1
        assert chunks[0]["message"]["content"] == "valid"

    def test_tools_included_in_payload_when_provided(self):
        lines = [json.dumps({"message": {"content": ""}, "done": True})]
        stream_ctx = make_stream_response(lines)
        instance = OllamaClient()
        instance._client = MagicMock()
        instance._client.stream.return_value = stream_ctx

        tools = [{"type": "function", "function": {"name": "shell"}}]
        list(instance.chat(model="test", messages=[], tools=tools))

        _, kwargs = instance._client.stream.call_args
        assert "tools" in kwargs["json"]
        assert kwargs["json"]["tools"] == tools

    def test_tools_omitted_when_none(self):
        lines = [json.dumps({"message": {"content": ""}, "done": True})]
        stream_ctx = make_stream_response(lines)
        instance = OllamaClient()
        instance._client = MagicMock()
        instance._client.stream.return_value = stream_ctx

        list(instance.chat(model="test", messages=[], tools=None))

        _, kwargs = instance._client.stream.call_args
        assert "tools" not in kwargs["json"]

    def test_correct_model_sent(self):
        lines = [json.dumps({"message": {"content": ""}, "done": True})]
        stream_ctx = make_stream_response(lines)
        instance = OllamaClient()
        instance._client = MagicMock()
        instance._client.stream.return_value = stream_ctx

        list(instance.chat(model="mymodel:latest", messages=[]))

        _, kwargs = instance._client.stream.call_args
        assert kwargs["json"]["model"] == "mymodel:latest"

    def test_messages_sent_in_payload(self):
        lines = [json.dumps({"message": {"content": ""}, "done": True})]
        stream_ctx = make_stream_response(lines)
        instance = OllamaClient()
        instance._client = MagicMock()
        instance._client.stream.return_value = stream_ctx

        msgs = [{"role": "user", "content": "hello"}]
        list(instance.chat(model="test", messages=msgs))

        _, kwargs = instance._client.stream.call_args
        assert kwargs["json"]["messages"] == msgs

    def test_hits_correct_endpoint(self):
        lines = [json.dumps({"message": {"content": ""}, "done": True})]
        stream_ctx = make_stream_response(lines)
        instance = OllamaClient("http://remote:5000")
        instance._client = MagicMock()
        instance._client.stream.return_value = stream_ctx

        list(instance.chat(model="test", messages=[]))

        args, _ = instance._client.stream.call_args
        assert args[1] == "http://remote:5000/api/chat"

    def test_tool_calls_in_chunk_preserved(self):
        tool_call = {"function": {"name": "shell", "arguments": {"command": "ls"}}}
        lines = [
            json.dumps({
                "message": {"role": "assistant", "content": "", "tool_calls": [tool_call]},
                "done": True,
            })
        ]
        stream_ctx = make_stream_response(lines)
        instance = OllamaClient()
        instance._client = MagicMock()
        instance._client.stream.return_value = stream_ctx

        chunks = list(instance.chat(model="test", messages=[]))
        assert chunks[0]["message"]["tool_calls"] == [tool_call]
