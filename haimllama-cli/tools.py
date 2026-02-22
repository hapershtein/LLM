"""Tool implementations for the agentic CLI."""

import subprocess
import os
import glob
import json
import re
import textwrap
from pathlib import Path
from typing import Any

import httpx


# ── Tool registry ──────────────────────────────────────────────────────────────

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "shell",
            "description": (
                "Run a shell command and return its stdout/stderr. "
                "Use for file ops, git, package managers, compiling, etc. "
                "Avoid long-running or interactive processes."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to execute.",
                    },
                    "cwd": {
                        "type": "string",
                        "description": "Working directory (optional, defaults to current dir).",
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Timeout in seconds (default 30).",
                    },
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file. Returns the text content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute or relative path to the file.",
                    },
                    "start_line": {
                        "type": "integer",
                        "description": "First line to read (1-indexed, optional).",
                    },
                    "end_line": {
                        "type": "integer",
                        "description": "Last line to read (1-indexed, optional).",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write or overwrite a file with given content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to write.",
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to write into the file.",
                    },
                    "append": {
                        "type": "boolean",
                        "description": "If true, append instead of overwrite (default false).",
                    },
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "List files and directories at a given path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory path to list (default '.').",
                    },
                    "show_hidden": {
                        "type": "boolean",
                        "description": "Include hidden files (default false).",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_files",
            "description": "Find files matching a glob pattern.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Glob pattern, e.g. '**/*.py' or 'src/*.ts'.",
                    },
                    "root": {
                        "type": "string",
                        "description": "Root directory to search from (default '.').",
                    },
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep",
            "description": "Search for a regex pattern inside files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Regex pattern to search for.",
                    },
                    "path": {
                        "type": "string",
                        "description": "File or directory to search in (default '.').",
                    },
                    "glob": {
                        "type": "string",
                        "description": "File glob filter, e.g. '*.py' (optional).",
                    },
                    "case_insensitive": {
                        "type": "boolean",
                        "description": "Case-insensitive matching (default false).",
                    },
                    "context_lines": {
                        "type": "integer",
                        "description": "Lines of context around each match (default 0).",
                    },
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "python_eval",
            "description": (
                "Execute Python code and return the output. "
                "Useful for calculations, data processing, and quick scripts. "
                "Use print() to produce output."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute.",
                    },
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_url",
            "description": "Fetch the content of a URL (HTTP GET). Returns text response.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "URL to fetch.",
                    },
                    "headers": {
                        "type": "object",
                        "description": "Optional HTTP headers as key-value pairs.",
                    },
                },
                "required": ["url"],
            },
        },
    },
]


# ── Implementations ────────────────────────────────────────────────────────────

def run_shell(command: str, cwd: str = None, timeout: int = 30) -> str:
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd or os.getcwd(),
        )
        output = ""
        if result.stdout:
            output += result.stdout
        if result.stderr:
            output += ("\n[stderr]\n" if result.stdout else "[stderr]\n") + result.stderr
        if not output:
            output = f"(exit code {result.returncode}, no output)"
        elif result.returncode != 0:
            output += f"\n[exit code {result.returncode}]"
        return output.rstrip()
    except subprocess.TimeoutExpired:
        return f"[error] Command timed out after {timeout}s"
    except Exception as e:
        return f"[error] {e}"


def read_file(path: str, start_line: int = None, end_line: int = None) -> str:
    try:
        p = Path(path).expanduser()
        if not p.exists():
            return f"[error] File not found: {path}"
        if p.stat().st_size > 2_000_000:
            return "[error] File too large (>2MB). Use start_line/end_line or grep."
        lines = p.read_text(errors="replace").splitlines()
        if start_line is not None or end_line is not None:
            s = (start_line or 1) - 1
            e = end_line or len(lines)
            lines = lines[s:e]
            prefix = start_line or 1
        else:
            prefix = 1
        numbered = "\n".join(f"{prefix + i:4}: {l}" for i, l in enumerate(lines))
        return numbered if numbered else "(empty file)"
    except Exception as e:
        return f"[error] {e}"


def write_file(path: str, content: str, append: bool = False) -> str:
    try:
        p = Path(path).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if append else "w"
        p.open(mode).write(content)
        action = "Appended" if append else "Written"
        return f"{action} {len(content)} chars to {path}"
    except Exception as e:
        return f"[error] {e}"


def list_dir(path: str = ".", show_hidden: bool = False) -> str:
    try:
        p = Path(path).expanduser()
        if not p.exists():
            return f"[error] Path not found: {path}"
        entries = sorted(p.iterdir(), key=lambda x: (x.is_file(), x.name.lower()))
        if not show_hidden:
            entries = [e for e in entries if not e.name.startswith(".")]
        if not entries:
            return "(empty directory)"
        lines = []
        for e in entries:
            if e.is_dir():
                lines.append(f"[DIR]  {e.name}/")
            elif e.is_symlink():
                lines.append(f"[LNK]  {e.name} -> {e.resolve()}")
            else:
                size = e.stat().st_size
                size_str = f"{size:>10,} B" if size < 1024 else (
                    f"{size/1024:>9.1f} KB" if size < 1_048_576 else f"{size/1_048_576:>9.1f} MB"
                )
                lines.append(f"[FILE] {e.name}  ({size_str})")
        return "\n".join(lines)
    except Exception as e:
        return f"[error] {e}"


def find_files(pattern: str, root: str = ".") -> str:
    try:
        root_path = Path(root).expanduser()
        matches = sorted(root_path.glob(pattern))
        if not matches:
            return f"No files matched '{pattern}' under {root}"
        return "\n".join(str(m) for m in matches[:200])
    except Exception as e:
        return f"[error] {e}"


def grep(
    pattern: str,
    path: str = ".",
    glob: str = None,
    case_insensitive: bool = False,
    context_lines: int = 0,
) -> str:
    try:
        flags = re.IGNORECASE if case_insensitive else 0
        rx = re.compile(pattern, flags)
        target = Path(path).expanduser()

        files: list[Path] = []
        if target.is_file():
            files = [target]
        else:
            file_glob = glob or "*"
            files = sorted(target.rglob(file_glob))

        results = []
        match_count = 0

        for fp in files:
            if not fp.is_file():
                continue
            try:
                text = fp.read_text(errors="replace")
            except Exception:
                continue
            lines = text.splitlines()
            for i, line in enumerate(lines):
                if rx.search(line):
                    match_count += 1
                    if match_count > 300:
                        results.append("... (truncated, >300 matches)")
                        return "\n".join(results)
                    ctx_start = max(0, i - context_lines)
                    ctx_end = min(len(lines), i + context_lines + 1)
                    block = []
                    for j in range(ctx_start, ctx_end):
                        marker = ">" if j == i else " "
                        block.append(f"{fp}:{j+1}{marker} {lines[j]}")
                    results.append("\n".join(block))

        if not results:
            return f"No matches for '{pattern}'"
        return "\n".join(results)
    except re.error as e:
        return f"[error] Invalid regex: {e}"
    except Exception as e:
        return f"[error] {e}"


def python_eval(code: str) -> str:
    import io
    import contextlib
    import traceback

    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            exec(textwrap.dedent(code), {})  # noqa: S102
        result = buf.getvalue()
        return result.rstrip() if result else "(no output)"
    except Exception:
        return traceback.format_exc()


def fetch_url(url: str, headers: dict = None) -> str:
    try:
        with httpx.Client(follow_redirects=True, timeout=15) as client:
            resp = client.get(url, headers=headers or {})
            content_type = resp.headers.get("content-type", "")
            text = resp.text
            if len(text) > 20_000:
                text = text[:20_000] + "\n...(truncated)"
            return f"[HTTP {resp.status_code}] {content_type}\n\n{text}"
    except Exception as e:
        return f"[error] {e}"


# ── Dispatch ───────────────────────────────────────────────────────────────────

TOOL_MAP: dict[str, Any] = {
    "shell": run_shell,
    "read_file": read_file,
    "write_file": write_file,
    "list_dir": list_dir,
    "find_files": find_files,
    "grep": grep,
    "python_eval": python_eval,
    "fetch_url": fetch_url,
}


def _coerce_types(arguments: dict) -> dict:
    """Coerce string 'true'/'false' to bool, numeric strings to int/float."""
    result = {}
    for k, v in arguments.items():
        if isinstance(v, str):
            if v.lower() == "true":
                result[k] = True
            elif v.lower() == "false":
                result[k] = False
            else:
                # Try int then float
                try:
                    result[k] = int(v)
                    continue
                except ValueError:
                    pass
                try:
                    result[k] = float(v)
                    continue
                except ValueError:
                    pass
                result[k] = v
        else:
            result[k] = v
    return result


def dispatch(name: str, arguments: dict) -> str:
    fn = TOOL_MAP.get(name)
    if fn is None:
        return f"[error] Unknown tool: {name}"
    try:
        return fn(**_coerce_types(arguments))
    except TypeError as e:
        return f"[error] Bad arguments for {name}: {e}"
