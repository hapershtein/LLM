"""Agentic loop: think → act → observe → repeat."""

import json
import re
from typing import Callable

from ollama_client import OllamaClient
from tools import TOOL_SCHEMAS, dispatch


def _sanitize_args(args: dict) -> dict:
    """
    Some models (e.g. llama3.2) double-serialize arguments so that a
    string parameter contains a JSON-encoded dict with the real value.
    This unwraps one level when detected.
    """
    cleaned = {}
    for k, v in args.items():
        if isinstance(v, str):
            stripped = v.strip()
            if stripped.startswith("{") or stripped.startswith("["):
                try:
                    parsed = json.loads(stripped)
                    # If parsed is a dict containing the same key, unwrap
                    if isinstance(parsed, dict) and k in parsed:
                        cleaned[k] = parsed[k]
                        continue
                except json.JSONDecodeError:
                    pass
        cleaned[k] = v
    return cleaned


_TEXT_TOOL_PATTERNS = [
    # <tool_call>{"name":"...", "arguments":{...}}</tool_call>
    re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL),
    # ```json\n{"name":"...", "arguments":{...}}\n```
    re.compile(r"```(?:json)?\s*(\{[^`]*\"name\"\s*:[^`]*\})\s*```", re.DOTALL),
    # bare JSON: {"name":"...", "arguments":{...}}
    re.compile(r'(\{"name"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:\s*\{[^}]*\}\s*\})', re.DOTALL),
]


def _extract_text_tool_calls(text: str) -> list[dict] | None:
    """
    For models that output tool calls as text rather than structured data,
    attempt to parse them out.  Returns None if nothing found.
    """
    for pat in _TEXT_TOOL_PATTERNS:
        matches = pat.findall(text)
        if matches:
            result = []
            for m in matches:
                try:
                    obj = json.loads(m)
                    if "name" in obj and "arguments" in obj:
                        result.append({
                            "function": {
                                "name": obj["name"],
                                "arguments": obj["arguments"],
                            }
                        })
                except json.JSONDecodeError:
                    continue
            if result:
                return result
    return None


class Agent:
    def __init__(
        self,
        model: str,
        client: OllamaClient,
        system_prompt: str = "",
        max_iterations: int = 20,
        on_tool_call: Callable[[str, dict], None] | None = None,
        on_tool_result: Callable[[str, str], None] | None = None,
        on_token: Callable[[str], None] | None = None,
    ):
        self.model = model
        self.client = client
        self.max_iterations = max_iterations
        self.on_tool_call = on_tool_call
        self.on_tool_result = on_tool_result
        self.on_token = on_token

        self.messages: list[dict] = []
        if system_prompt:
            self.messages.append({"role": "system", "content": system_prompt})

    def _run_tool_calls(self, tool_calls: list[dict]) -> list[dict]:
        """Execute tool calls and return tool-result messages."""
        result_messages = []
        for tc in tool_calls:
            name = tc.get("function", {}).get("name", "")
            raw_args = tc.get("function", {}).get("arguments", {})

            # Arguments may come as a JSON string or already a dict
            if isinstance(raw_args, str):
                try:
                    args = json.loads(raw_args)
                except json.JSONDecodeError:
                    args = {}
            else:
                args = raw_args

            args = _sanitize_args(args)

            if self.on_tool_call:
                self.on_tool_call(name, args)

            result = dispatch(name, args)

            if self.on_tool_result:
                self.on_tool_result(name, result)

            result_messages.append({
                "role": "tool",
                "content": result,
            })
        return result_messages

    def run(self, user_message: str) -> str:
        """Add user message and run the agentic loop until done."""
        self.messages.append({"role": "user", "content": user_message})

        for iteration in range(self.max_iterations):
            # Use streaming only when we expect a text response (no tools needed
            # yet), but Ollama returns tool_calls in the final done chunk.
            # We collect the full response first, then check for tool_calls.
            assistant_content = ""
            tool_calls: list[dict] = []
            finish_reason = None

            for chunk in self.client.chat(
                model=self.model,
                messages=self.messages,
                tools=TOOL_SCHEMAS,
                stream=True,
            ):
                msg = chunk.get("message", {})
                delta_content = msg.get("content", "")
                delta_tools = msg.get("tool_calls", [])

                if delta_content:
                    assistant_content += delta_content
                    # Only stream tokens when no tool calls are accumulating
                    if not tool_calls and self.on_token and delta_content:
                        self.on_token(delta_content)

                if delta_tools:
                    tool_calls.extend(delta_tools)

                if chunk.get("done"):
                    finish_reason = chunk.get("done_reason", "stop")
                    break

            # Build the assistant message to append
            assistant_msg: dict = {"role": "assistant", "content": assistant_content}
            if tool_calls:
                assistant_msg["tool_calls"] = tool_calls

            self.messages.append(assistant_msg)

            # If there are tool calls, execute them and loop
            if tool_calls:
                result_messages = self._run_tool_calls(tool_calls)
                self.messages.extend(result_messages)
                continue

            # Fallback: model may output tool calls as text (no native support)
            if assistant_content:
                text_calls = _extract_text_tool_calls(assistant_content)
                if text_calls:
                    result_messages = self._run_tool_calls(text_calls)
                    self.messages.extend(result_messages)
                    continue

            # No tool calls → final answer
            return assistant_content

        return "[max iterations reached]"

    def clear(self):
        """Clear conversation history (keep system prompt)."""
        self.messages = [m for m in self.messages if m["role"] == "system"]
