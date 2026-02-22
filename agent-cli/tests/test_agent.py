"""Unit tests for agent.py — Agent class and helper functions."""

import sys
import json
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent import Agent, _sanitize_args, _extract_text_tool_calls


# ── Helpers ────────────────────────────────────────────────────────────────────

def make_chunk(content="", tool_calls=None, done=False, done_reason="stop"):
    """Build a fake Ollama streaming chunk."""
    msg = {"role": "assistant", "content": content}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return {"message": msg, "done": done, "done_reason": done_reason}


def mock_client(chunks: list[dict]):
    """Return a fake OllamaClient whose .chat() yields the given chunks."""
    client = MagicMock()
    client.chat.return_value = iter(chunks)
    return client


# ── _sanitize_args ─────────────────────────────────────────────────────────────

class TestSanitizeArgs:
    def test_plain_string_unchanged(self):
        args = {"command": "echo hi"}
        assert _sanitize_args(args) == {"command": "echo hi"}

    def test_double_serialized_string_unwrapped(self):
        # llama3.2 quirk: {"code": "{\"code\":\"print(1)\"}"}
        args = {"code": '{"code": "print(1)"}'}
        result = _sanitize_args(args)
        assert result["code"] == "print(1)"

    def test_nested_dict_not_unwrapped_wrongly(self):
        # If the inner dict has a different key, don't unwrap
        args = {"code": '{"other_key": "value"}'}
        result = _sanitize_args(args)
        # Should remain unchanged because "code" key not found inside
        assert result["code"] == '{"other_key": "value"}'

    def test_non_string_value_unchanged(self):
        args = {"timeout": 30, "flag": True}
        assert _sanitize_args(args) == {"timeout": 30, "flag": True}

    def test_empty_dict(self):
        assert _sanitize_args({}) == {}

    def test_invalid_json_string_unchanged(self):
        args = {"code": "{not valid json"}
        assert _sanitize_args(args) == {"code": "{not valid json"}

    def test_list_value_not_unwrapped(self):
        args = {"items": '["a", "b"]'}
        result = _sanitize_args(args)
        assert result["items"] == '["a", "b"]'

    def test_multiple_keys_each_processed(self):
        args = {
            "code": '{"code": "x = 1"}',
            "other": "plain",
        }
        result = _sanitize_args(args)
        assert result["code"] == "x = 1"
        assert result["other"] == "plain"


# ── _extract_text_tool_calls ───────────────────────────────────────────────────

class TestExtractTextToolCalls:
    def test_xml_tag_format(self):
        text = '<tool_call>{"name": "shell", "arguments": {"command": "ls"}}</tool_call>'
        result = _extract_text_tool_calls(text)
        assert result is not None
        assert result[0]["function"]["name"] == "shell"
        assert result[0]["function"]["arguments"] == {"command": "ls"}

    def test_markdown_fence_format(self):
        text = '```json\n{"name": "python_eval", "arguments": {"code": "print(1)"}}\n```'
        result = _extract_text_tool_calls(text)
        assert result is not None
        assert result[0]["function"]["name"] == "python_eval"

    def test_bare_json_format(self):
        text = '{"name": "list_dir", "arguments": {"path": "/tmp"}}'
        result = _extract_text_tool_calls(text)
        assert result is not None
        assert result[0]["function"]["name"] == "list_dir"

    def test_no_tool_call_returns_none(self):
        text = "This is just a plain response with no tool calls."
        assert _extract_text_tool_calls(text) is None

    def test_incomplete_json_ignored(self):
        text = '{"name": "shell", "arguments": {"command":'
        assert _extract_text_tool_calls(text) is None

    def test_multiple_tool_calls_extracted(self):
        text = (
            '<tool_call>{"name": "shell", "arguments": {"command": "pwd"}}</tool_call>\n'
            '<tool_call>{"name": "shell", "arguments": {"command": "ls"}}</tool_call>'
        )
        result = _extract_text_tool_calls(text)
        assert result is not None
        assert len(result) == 2

    def test_object_without_name_key_ignored(self):
        text = '{"foo": "bar", "arguments": {"x": 1}}'
        assert _extract_text_tool_calls(text) is None


# ── Agent.__init__ ─────────────────────────────────────────────────────────────

class TestAgentInit:
    def test_no_system_prompt(self):
        client = mock_client([])
        agent = Agent(model="test", client=client)
        assert agent.messages == []

    def test_with_system_prompt(self):
        client = mock_client([])
        agent = Agent(model="test", client=client, system_prompt="You are helpful.")
        assert len(agent.messages) == 1
        assert agent.messages[0]["role"] == "system"
        assert agent.messages[0]["content"] == "You are helpful."

    def test_default_max_iterations(self):
        client = mock_client([])
        agent = Agent(model="test", client=client)
        assert agent.max_iterations == 20

    def test_custom_max_iterations(self):
        client = mock_client([])
        agent = Agent(model="test", client=client, max_iterations=5)
        assert agent.max_iterations == 5


# ── Agent.clear ────────────────────────────────────────────────────────────────

class TestAgentClear:
    def test_clears_user_and_assistant_messages(self):
        client = mock_client([])
        agent = Agent(model="test", client=client, system_prompt="sys")
        agent.messages.append({"role": "user", "content": "hi"})
        agent.messages.append({"role": "assistant", "content": "hello"})
        agent.clear()
        assert len(agent.messages) == 1
        assert agent.messages[0]["role"] == "system"

    def test_clear_without_system_prompt(self):
        client = mock_client([])
        agent = Agent(model="test", client=client)
        agent.messages.append({"role": "user", "content": "hi"})
        agent.clear()
        assert agent.messages == []

    def test_clear_empty_history_is_safe(self):
        client = mock_client([])
        agent = Agent(model="test", client=client)
        agent.clear()  # Should not raise
        assert agent.messages == []


# ── Agent._run_tool_calls ──────────────────────────────────────────────────────

class TestAgentRunToolCalls:
    def test_dispatches_tool_and_returns_result_message(self):
        client = mock_client([])
        agent = Agent(model="test", client=client)
        tool_calls = [{"function": {"name": "shell", "arguments": {"command": "echo hi"}}}]

        results = agent._run_tool_calls(tool_calls)

        assert len(results) == 1
        assert results[0]["role"] == "tool"
        assert "hi" in results[0]["content"]

    def test_fires_on_tool_call_callback(self):
        client = mock_client([])
        called_with = []
        agent = Agent(model="test", client=client, on_tool_call=lambda n, a: called_with.append((n, a)))

        agent._run_tool_calls([{"function": {"name": "shell", "arguments": {"command": "echo x"}}}])

        assert len(called_with) == 1
        assert called_with[0][0] == "shell"

    def test_fires_on_tool_result_callback(self):
        client = mock_client([])
        results_seen = []
        agent = Agent(model="test", client=client, on_tool_result=lambda n, r: results_seen.append((n, r)))

        agent._run_tool_calls([{"function": {"name": "python_eval", "arguments": {"code": "print(99)"}}}])

        assert len(results_seen) == 1
        assert results_seen[0][0] == "python_eval"
        assert "99" in results_seen[0][1]

    def test_json_string_args_parsed(self):
        client = mock_client([])
        agent = Agent(model="test", client=client)
        # arguments as JSON string (some models do this)
        tool_calls = [{"function": {"name": "python_eval", "arguments": '{"code": "print(7)"}'}}]

        results = agent._run_tool_calls(tool_calls)
        assert "7" in results[0]["content"]

    def test_unknown_tool_returns_error_message(self):
        client = mock_client([])
        agent = Agent(model="test", client=client)
        tool_calls = [{"function": {"name": "does_not_exist", "arguments": {}}}]

        results = agent._run_tool_calls(tool_calls)
        assert "[error]" in results[0]["content"]

    def test_multiple_tool_calls_all_executed(self):
        client = mock_client([])
        agent = Agent(model="test", client=client)
        tool_calls = [
            {"function": {"name": "python_eval", "arguments": {"code": "print('a')"}}},
            {"function": {"name": "python_eval", "arguments": {"code": "print('b')"}}},
        ]
        results = agent._run_tool_calls(tool_calls)
        assert len(results) == 2
        assert "a" in results[0]["content"]
        assert "b" in results[1]["content"]


# ── Agent.run — no tool calls ──────────────────────────────────────────────────

class TestAgentRunNoTools:
    def test_returns_content_on_final_answer(self):
        chunks = [
            make_chunk(content="Hello "),
            make_chunk(content="world", done=True),
        ]
        client = mock_client(chunks)
        agent = Agent(model="test", client=client)

        result = agent.run("hi")
        assert result == "Hello world"

    def test_user_message_appended(self):
        chunks = [make_chunk(content="ok", done=True)]
        client = mock_client(chunks)
        agent = Agent(model="test", client=client)

        agent.run("test message")
        assert agent.messages[0]["role"] == "user"
        assert agent.messages[0]["content"] == "test message"

    def test_assistant_message_appended(self):
        chunks = [make_chunk(content="response", done=True)]
        client = mock_client(chunks)
        agent = Agent(model="test", client=client)

        agent.run("q")
        assert agent.messages[-1]["role"] == "assistant"
        assert agent.messages[-1]["content"] == "response"

    def test_on_token_callback_fires(self):
        chunks = [
            make_chunk(content="tok1"),
            make_chunk(content="tok2", done=True),
        ]
        client = mock_client(chunks)
        tokens = []
        agent = Agent(model="test", client=client, on_token=tokens.append)

        agent.run("go")
        assert "tok1" in tokens
        assert "tok2" in tokens

    def test_conversation_accumulates(self):
        def make_chunks(text):
            return iter([make_chunk(content=text, done=True)])

        client = MagicMock()
        client.chat.side_effect = [
            make_chunks("first"),
            make_chunks("second"),
        ]
        agent = Agent(model="test", client=client)

        agent.run("q1")
        agent.run("q2")

        roles = [m["role"] for m in agent.messages]
        assert roles == ["user", "assistant", "user", "assistant"]


# ── Agent.run — native tool calls ─────────────────────────────────────────────

class TestAgentRunNativeToolCalls:
    def test_executes_tool_then_returns_final_answer(self):
        tool_call = {
            "id": "call_1",
            "function": {"name": "python_eval", "arguments": {"code": "print(42)"}},
        }
        # First call: tool_calls in message; Second call: final text answer
        chunks_1 = [make_chunk(tool_calls=[tool_call], done=True)]
        chunks_2 = [make_chunk(content="The answer is 42", done=True)]

        client = MagicMock()
        client.chat.side_effect = [iter(chunks_1), iter(chunks_2)]
        agent = Agent(model="test", client=client)

        result = agent.run("compute 42")
        assert "42" in result

    def test_tool_result_added_to_messages(self):
        tool_call = {"function": {"name": "python_eval", "arguments": {"code": "print(1)"}}}
        chunks_1 = [make_chunk(tool_calls=[tool_call], done=True)]
        chunks_2 = [make_chunk(content="done", done=True)]

        client = MagicMock()
        client.chat.side_effect = [iter(chunks_1), iter(chunks_2)]
        agent = Agent(model="test", client=client)

        agent.run("test")
        tool_msgs = [m for m in agent.messages if m["role"] == "tool"]
        assert len(tool_msgs) == 1

    def test_token_callback_suppressed_after_tool_calls_seen(self):
        # Tokens in a chunk that arrives AFTER tool_calls have already accumulated
        # in the current turn should be suppressed.
        tool_call = {"function": {"name": "python_eval", "arguments": {"code": "print(5)"}}}
        # Chunk 1: tool_calls only (no content yet)
        # Chunk 2: content arrives after tool_calls have already been registered
        chunks_1 = [
            make_chunk(tool_calls=[tool_call], done=False),
            make_chunk(content="late token", done=True),
        ]
        chunks_2 = [make_chunk(content="final", done=True)]

        client = MagicMock()
        client.chat.side_effect = [iter(chunks_1), iter(chunks_2)]
        tokens = []
        agent = Agent(model="test", client=client, on_token=tokens.append)

        agent.run("go")
        # "late token" arrives after tool_calls were seen → suppressed
        assert "late token" not in tokens
        # "final" is the answer after tools ran → should stream
        assert "final" in tokens


# ── Agent.run — text-based tool call fallback ─────────────────────────────────

class TestAgentRunTextFallback:
    def test_text_tool_call_executed(self):
        # Model emits tool call as plain JSON text (no native support)
        tool_json = '{"name": "python_eval", "arguments": {"code": "print(99)"}}'
        chunks_1 = [make_chunk(content=tool_json, done=True)]
        chunks_2 = [make_chunk(content="done", done=True)]

        client = MagicMock()
        client.chat.side_effect = [iter(chunks_1), iter(chunks_2)]
        agent = Agent(model="test", client=client)

        result = agent.run("compute")
        assert "done" in result

    def test_xml_format_fallback(self):
        tool_xml = '<tool_call>{"name": "shell", "arguments": {"command": "echo works"}}</tool_call>'
        chunks_1 = [make_chunk(content=tool_xml, done=True)]
        chunks_2 = [make_chunk(content="all good", done=True)]

        client = MagicMock()
        client.chat.side_effect = [iter(chunks_1), iter(chunks_2)]
        agent = Agent(model="test", client=client)

        result = agent.run("test")
        assert "all good" in result


# ── Agent.run — max iterations ────────────────────────────────────────────────

class TestAgentRunMaxIterations:
    def test_stops_at_max_iterations(self):
        # Every response contains a tool call, forcing endless loops
        tool_call = {"function": {"name": "python_eval", "arguments": {"code": "print(1)"}}}

        def infinite_chunks():
            while True:
                yield make_chunk(tool_calls=[tool_call], done=True)

        client = MagicMock()
        client.chat.side_effect = lambda **_: infinite_chunks()

        agent = Agent(model="test", client=client, max_iterations=3)
        result = agent.run("loop forever")
        assert "max iterations" in result
