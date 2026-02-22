# Tasks — Agentic CLI

Track of all planned, in-progress, and completed work.
Format: `[status]` — `description` — `notes`

Legend: `[x]` done · `[~]` in progress · `[ ]` todo · `[!]` blocked

---

## Milestone 0 — Foundation ✅

- [x] **M0-1** Bootstrap project structure (`main.py`, `agent.py`, `tools.py`, `ollama_client.py`)
- [x] **M0-2** Implement Ollama streaming client (`/api/chat` with `stream: true`)
- [x] **M0-3** Implement ReAct agentic loop (native tool call support)
- [x] **M0-4** Implement all 8 built-in tools (shell, read_file, write_file, list_dir, find_files, grep, python_eval, fetch_url)
- [x] **M0-5** Rich terminal UI with streaming tokens + tool call panels
- [x] **M0-6** Interactive REPL with `/help`, `/clear`, `/model`, `/models`, `/tools`, `/history`, `/save`, `/load`, `/exit`
- [x] **M0-7** One-shot mode and piped stdin support
- [x] **M0-8** `--model`, `--url`, `--no-tools`, `--max-iter`, `--list-models` CLI flags
- [x] **M0-9** Config persistence (`~/.config/agent-cli/config.json`)
- [x] **M0-10** Bash launcher script (`agent-cli`)

---

## Milestone 1 — Robustness ✅

- [x] **M1-1** Argument type coercion (`"true"` → `True`, `"42"` → `42`) in tool dispatch
- [x] **M1-2** Double-serialized argument sanitization (llama3.2 quirk)
- [x] **M1-3** Text-based tool call fallback for models without native tool support
  - Handles: `<tool_call>{...}</tool_call>`, ` ```json {...}` ` `, bare JSON objects
- [x] **M1-4** Tool result truncation (2000 chars in UI, 20KB for fetch_url)
- [x] **M1-5** Graceful Ctrl+C / Ctrl+D handling in REPL

---

## Milestone 2 — Tests ✅

- [x] **M2-1** Unit tests for all 8 tools (`tests/test_tools.py`)
  - run_shell: basic, stderr, exit codes, timeout, cwd
  - read_file: content, line ranges, missing file, large file
  - write_file: create, overwrite, append, nested dirs
  - list_dir: files, dirs, hidden toggle, empty
  - find_files: glob matching, recursive, no match
  - grep: pattern, case sensitivity, context lines, glob filter, invalid regex
  - python_eval: print, arithmetic, imports, errors, dedent, stderr
  - fetch_url: success, truncation, connection error (mocked)
  - dispatch: known/unknown tools, bad args, type coercion
  - _coerce_types: all type conversions
  - TOOL_SCHEMAS: schema validation
- [x] **M2-2** Unit tests for agent logic (`tests/test_agent.py`)
  - _sanitize_args: unwrapping, edge cases
  - _extract_text_tool_calls: all 3 formats + no-match + multi-call
  - Agent init, clear, message accumulation
  - _run_tool_calls: dispatch, callbacks, multiple calls
  - Agent.run: no-tools, streaming, native tool calls, text fallback, max iterations
- [x] **M2-3** Unit tests for Ollama client (`tests/test_ollama_client.py`)
  - list_models: success, empty, missing key, connection error, endpoint URL
  - chat: chunk yielding, done-stop, empty/invalid lines, payload construction, tool_calls
- [x] **M2-4** Write PRD.md
- [x] **M2-5** Write TASKS.md

---

## Milestone 3 — Quality [ ]

- [ ] **M3-1** Add `pytest-cov` and enforce ≥ 80% coverage gate in tests
- [ ] **M3-2** Add `pyproject.toml` / `setup.py` for `pip install -e .` support
- [ ] **M3-3** Add `--verbose` / `--debug` flag to dump raw Ollama responses
- [ ] **M3-4** Validate tool schemas against JSON Schema spec at startup
- [ ] **M3-5** Add integration test that hits a real Ollama instance (pytest mark: `slow`)
- [ ] **M3-6** Lint with `ruff` / `mypy` (add to CI)

---

## Milestone 4 — Features [ ]

- [ ] **M4-1** **Tool: `edit_file`** — patch/replace a line range in an existing file
- [ ] **M4-2** **Tool: `web_search`** — DuckDuckGo or SearXNG search (local instance)
- [ ] **M4-3** **`/retry`** REPL command — re-run last user message with fresh context
- [ ] **M4-4** **`/undo`** REPL command — pop last exchange from history
- [ ] **M4-5** **System prompt templates** — `--persona coder|analyst|sysadmin`
- [ ] **M4-6** **Multi-turn context window management** — auto-summarise old messages when approaching model context limit
- [ ] **M4-7** **Token usage display** — show prompt/eval tokens after each response

---

## Milestone 5 — Security & Safety [ ]

- [ ] **M5-1** **Confirmation prompts** for destructive shell commands (`rm`, `dd`, `mkfs`, etc.)
- [ ] **M5-2** **Sandboxed `python_eval`** via `subprocess` with resource limits
- [ ] **M5-3** **Allowlist/denylist** for shell commands (configurable in `config.json`)
- [ ] **M5-4** **Read-only mode** flag (`--read-only`) — disables write_file, shell writes

---

## Milestone 6 — Roadmap [ ]

- [ ] **M6-1** Persistent memory with local embeddings (RAG over past conversations)
- [ ] **M6-2** Custom tool plugins from `~/.config/agent-cli/tools/*.py`
- [ ] **M6-3** Multi-agent: spawn subagents for parallel subtasks
- [ ] **M6-4** Vision: pass images to multimodal models (llava, llama3.2-vision)
- [ ] **M6-5** TUI dashboard with split-pane layout (chat | tool log | file tree)
- [ ] **M6-6** Named sessions with auto save/restore

---

## Bugs / Known Issues

| ID | Description | Status |
|----|-------------|--------|
| B1 | `qwen2.5-coder` outputs `show_hidden: "false"` as string → now handled by `_coerce_types` | Fixed |
| B2 | `llama3.2` double-serializes some arguments → now handled by `_sanitize_args` | Fixed |
| B3 | `qwen2.5max` is very slow to respond (>90s for first token) | Open — model-specific |
| B4 | Token streaming suppressed when model emits content alongside tool_calls | By design |
