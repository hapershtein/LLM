# Product Requirements Document — Agentic CLI

**Version:** 1.0
**Date:** 2026-02-22
**Status:** Active Development

---

## 1. Overview

Agentic CLI is a local, offline-first AI agent that runs in the terminal. It connects to
[Ollama](https://ollama.com) to use open-source language models and can autonomously use
tools to complete multi-step tasks — reading and writing files, running shell commands,
executing Python, searching codebases, and fetching URLs — without sending data to any
external service.

---

## 2. Problem Statement

Developers and power users need an AI assistant that:

- **Works offline / on-premises** — no API keys, no data leaving the machine.
- **Can act, not just answer** — reading files, running code, and making changes to complete tasks end-to-end.
- **Integrates with the terminal** — where most development work already happens.
- **Is model-agnostic** — works with any model available in Ollama.

Existing solutions either require cloud connectivity, lack tool use, or are GUI-only.

---

## 3. Goals

| # | Goal |
|---|------|
| G1 | Work entirely offline using locally-hosted Ollama models |
| G2 | Support an agentic ReAct loop (reason → act → observe → repeat) |
| G3 | Provide a rich, ergonomic terminal interface with streaming output |
| G4 | Be model-agnostic; degrade gracefully for models without native tool support |
| G5 | Support both interactive REPL and scriptable one-shot / piped usage |
| G6 | Have a clear, extensible architecture for adding new tools |

---

## 4. Non-Goals

- GUI or web interface
- Multi-user / server mode
- Plugin marketplace or remote tool execution
- Fine-tuning or model management (deferred to Ollama)
- Sandboxed code execution (out of scope for v1)

---

## 5. Users

**Primary:** Developers and engineers who work heavily in the terminal and want an AI coding/ops assistant that runs locally.

**Secondary:** Privacy-conscious users who need AI assistance without cloud connectivity; homelab and self-hosted AI enthusiasts.

---

## 6. Features

### 6.1 Core — Agentic Loop

| ID | Feature | Priority |
|----|---------|----------|
| F1 | ReAct loop: model reasons, calls tools, observes results, repeats | P0 |
| F2 | Native Ollama tool-calling API support | P0 |
| F3 | Text-based tool call fallback for models without native support | P0 |
| F4 | Argument sanitization (handles double-serialized args from llama3.2) | P0 |
| F5 | Configurable max iteration limit (default 20, `--max-iter`) | P1 |

### 6.2 Built-in Tools

| ID | Tool | Description | Priority |
|----|------|-------------|----------|
| T1 | `shell` | Execute shell commands with timeout, cwd, stderr capture | P0 |
| T2 | `read_file` | Read file contents with optional line range | P0 |
| T3 | `write_file` | Write or append to files, auto-create parent dirs | P0 |
| T4 | `list_dir` | List directory contents with sizes, hidden file toggle | P0 |
| T5 | `find_files` | Glob pattern file search | P0 |
| T6 | `grep` | Regex search across files with context lines | P0 |
| T7 | `python_eval` | Execute Python code, capture stdout/stderr | P0 |
| T8 | `fetch_url` | HTTP GET with truncation for large responses | P1 |

### 6.3 CLI Interface

| ID | Feature | Priority |
|----|---------|----------|
| F6 | Interactive REPL with Rich-formatted panels | P0 |
| F7 | One-shot mode: `python main.py "query"` | P0 |
| F8 | Piped stdin: `echo "..." \| python main.py` | P0 |
| F9 | `--model` / `-m` flag to select Ollama model | P0 |
| F10 | `--list-models` to enumerate available models | P1 |
| F11 | `--no-tools` flag for plain chat mode | P1 |
| F12 | `--url` flag for remote Ollama instances | P1 |

### 6.4 REPL Commands

| ID | Command | Priority |
|----|---------|----------|
| F13 | `/help` — show command reference | P0 |
| F14 | `/clear` — reset conversation history | P0 |
| F15 | `/model <name>` — switch model mid-session | P1 |
| F16 | `/models` — list available Ollama models | P1 |
| F17 | `/tools` — list available tools | P1 |
| F18 | `/history` — show conversation messages | P2 |
| F19 | `/save <file>` — persist conversation to JSON | P2 |
| F20 | `/load <file>` — restore conversation from JSON | P2 |

### 6.5 UX & Output

| ID | Feature | Priority |
|----|---------|----------|
| F21 | Streaming token output (live text as model generates) | P0 |
| F22 | Rich panel for each tool call (yellow border, JSON args) | P1 |
| F23 | Rich panel for each tool result (green border, syntax-highlighted) | P1 |
| F24 | Markdown rendering for final assistant responses | P1 |
| F25 | Graceful handling of Ctrl+C / Ctrl+D | P0 |

### 6.6 Configuration

| ID | Feature | Priority |
|----|---------|----------|
| F26 | Persist last-used model to `~/.config/agent-cli/config.json` | P2 |
| F27 | `OLLAMA_URL` env var for base URL override | P1 |

---

## 7. Architecture

```
main.py          CLI entry point, REPL loop, UI rendering
  └── agent.py   Agentic loop (ReAct), message history, tool dispatch
        ├── ollama_client.py   Streaming HTTP client for Ollama /api/chat
        └── tools.py           Tool schemas (JSON) + implementations
```

**Key design decisions:**

- **No external agent framework** — lightweight, dependency-minimal implementation.
- **Streaming by default** — tool calls surface in the final "done" chunk; tokens stream to terminal live.
- **Graceful degradation** — if a model outputs tool calls as text JSON instead of structured API data, a regex-based fallback parses them.
- **Type coercion in dispatch** — string `"true"/"false"` and numeric strings are coerced before being passed to tool functions, tolerating imprecise model output.

---

## 8. Technical Constraints

- Python 3.12+
- Dependencies: `httpx` (HTTP client), `rich` (terminal UI)
- Requires Ollama running locally (or accessible at `OLLAMA_URL`)
- No root/sudo needed; writes only to `~/.config/agent-cli/`

---

## 9. Success Metrics

| Metric | Target |
|--------|--------|
| Tool round-trip latency overhead | < 50ms per tool call (excluding model inference) |
| Models supported out of the box | All Ollama models; full tool use on llama3.2+, qwen2.5+ |
| Test coverage | ≥ 80% across tools, agent loop, and client |
| Cold start (import + first prompt) | < 2s on local hardware |

---

## 10. Future Roadmap (v2+)

- [ ] **Sandboxed execution** — Docker/Podman container for `shell` and `python_eval`
- [ ] **Persistent memory** — RAG over conversation history using local embeddings
- [ ] **Custom tools** — load tool plugins from `~/.config/agent-cli/tools/`
- [ ] **Multi-agent** — spawn subagents for parallelisable subtasks
- [ ] **Vision support** — pass screenshots/images to multimodal models (llava)
- [ ] **TUI dashboard** — split-pane view: chat | tool call log | file tree
- [ ] **Session management** — named sessions with automatic save/restore
