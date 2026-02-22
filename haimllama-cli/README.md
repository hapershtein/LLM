# haimllama-cli

A local agentic CLI powered by [Ollama](https://ollama.com). The agent can reason, plan, and use tools to complete tasks — all running offline on your machine.

## Setup

```bash
pip install httpx rich
```

## Usage

```bash
# Interactive REPL (default model: llama3.2)
python main.py

# Pick a specific model
python main.py -m qwen2.5-coder:latest

# One-shot query
python main.py "write a Python script that counts words in a file"

# Pipe stdin
cat mycode.py | python main.py "review this code for bugs"

# List available models
python main.py --list-models

# Disable tools (plain chat)
python main.py --no-tools
```

## REPL Commands

| Command | Description |
|---------|-------------|
| `/help` | Show help |
| `/tools` | List available tools |
| `/clear` | Clear conversation history |
| `/model <name>` | Switch model |
| `/models` | List Ollama models |
| `/history` | Show message history |
| `/save <file>` | Save conversation to JSON |
| `/load <file>` | Load conversation from JSON |
| `/exit` | Quit |

## Tools

| Tool | Description |
|------|-------------|
| `shell` | Run shell commands |
| `read_file` | Read file contents |
| `write_file` | Write/create files |
| `list_dir` | List directory contents |
| `find_files` | Find files by glob pattern |
| `grep` | Search file contents with regex |
| `python_eval` | Execute Python code |
| `fetch_url` | HTTP GET a URL |

## Models

Best models for tool use (tested):
- `llama3.2:latest` — native tool calling, fast
- `qwen2.5:latest` — good tool support
- `qwen2.5-coder:latest` — great for code (text-based tool fallback)

## Config

Settings saved to `~/.config/haimllama-cli/config.json` (last used model).

Set `OLLAMA_URL` env var to use a remote Ollama instance.
