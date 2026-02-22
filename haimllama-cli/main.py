#!/usr/bin/env python3
"""
haimllama-cli — a local AI agent powered by Ollama.

Usage:
    python main.py                        # interactive mode, default model
    python main.py -m qwen2.5-coder      # pick a model
    python main.py -m llama3.2 --no-tools  # disable tool use
    python main.py "What files are in /tmp?"  # single-shot query
    echo "summarise this" | python main.py   # pipe stdin
"""

import argparse
import json
import os
import sys
import textwrap
from pathlib import Path

import readline  # noqa: F401 — enables arrow keys, backspace, history for input()

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich import print as rprint

from agent import Agent
from ollama_client import OllamaClient
from tools import TOOL_MAP, TOOL_SCHEMAS

# ── Constants ──────────────────────────────────────────────────────────────────

DEFAULT_MODEL = "qwen2.5:7b"
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
CONFIG_PATH = Path.home() / ".config" / "haimllama-cli" / "config.json"

SYSTEM_PROMPT = """You are a highly capable local AI agent. You have access to tools that let you:
- Run shell commands
- Read, write, and search files
- Execute Python code
- Fetch URLs

Always use the minimum tools necessary. Think step-by-step.
When writing code or files, verify the result. Be concise but thorough.
If a task is ambiguous, clarify before acting.
Current working directory: {cwd}
"""

HELP_TEXT = """
[bold cyan]haimllama-cli — Commands[/bold cyan]

  [bold]/help[/bold]            Show this help
  [bold]/tools[/bold]           List available tools
  [bold]/clear[/bold]           Clear conversation history
  [bold]/model[/bold] [dim]<name>[/dim]    Switch model
  [bold]/models[/bold]          List available Ollama models
  [bold]/history[/bold]         Show conversation history
  [bold]/save[/bold] [dim]<file>[/dim]     Save conversation to JSON file
  [bold]/load[/bold] [dim]<file>[/dim]     Load conversation from JSON file
  [bold]/exit[/bold]            Exit (also: Ctrl+C, Ctrl+D)

[bold]Tips:[/bold]
  • Pipe input:  echo "what files are here?" | haimllama-cli
  • One-shot:    haimllama-cli "write me a hello world in Rust"
  • Pick model:  haimllama-cli -m llama3.2
"""


# ── UI helpers ─────────────────────────────────────────────────────────────────

console = Console()


def print_tool_call(name: str, args: dict) -> None:
    args_str = json.dumps(args, indent=2) if args else "{}"
    console.print(
        Panel(
            Syntax(args_str, "json", theme="monokai", word_wrap=True),
            title=f"[bold yellow]⚙ tool:[/bold yellow] [cyan]{name}[/cyan]",
            border_style="yellow",
            expand=False,
        )
    )


def print_tool_result(name: str, result: str) -> None:
    MAX = 2000
    truncated = result
    suffix = ""
    if len(result) > MAX:
        truncated = result[:MAX]
        suffix = f"\n[dim]... ({len(result) - MAX} chars truncated)[/dim]"

    # Try to syntax-highlight if it looks like code
    lang = "text"
    stripped = truncated.strip()
    if stripped.startswith("{") or stripped.startswith("["):
        lang = "json"
    elif any(stripped.startswith(kw) for kw in ("def ", "class ", "import ", "from ", "#!")):
        lang = "python"

    console.print(
        Panel(
            Syntax(truncated, lang, theme="monokai", word_wrap=True),
            title=f"[bold green]✓ result:[/bold green] [cyan]{name}[/cyan]{suffix}",
            border_style="green",
            expand=False,
        )
    )


def print_assistant(text: str) -> None:
    console.print()
    console.print(Markdown(text))
    console.print()


def stream_token(token: str) -> None:
    console.print(token, end="", markup=False)


# ── Config persistence ─────────────────────────────────────────────────────────

def load_config() -> dict:
    if CONFIG_PATH.exists():
        try:
            return json.loads(CONFIG_PATH.read_text())
        except Exception:
            pass
    return {}


def save_config(cfg: dict) -> None:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2))


# ── REPL ───────────────────────────────────────────────────────────────────────

def repl(agent: Agent, client: OllamaClient, model_holder: list[str]) -> None:
    """Interactive REPL loop."""
    streaming_active = False  # track if we're mid-stream

    def on_token(tok: str):
        nonlocal streaming_active
        if not streaming_active:
            console.print()
            console.print("[bold blue]Assistant[/bold blue] ", end="")
            streaming_active = True
        console.print(tok, end="", markup=False)

    def finish_stream():
        nonlocal streaming_active
        if streaming_active:
            console.print()
            streaming_active = False

    agent.on_token = on_token
    agent.on_tool_call = lambda n, a: (finish_stream(), print_tool_call(n, a))
    agent.on_tool_result = lambda n, r: print_tool_result(n, r)

    console.print(
        Panel(
            f"[bold]Model:[/bold] [cyan]{model_holder[0]}[/cyan]  "
            f"[bold]Tools:[/bold] {len(TOOL_SCHEMAS)} available\n"
            f"[dim]Type [bold]/help[/bold] for commands, [bold]/exit[/bold] to quit[/dim]",
            title="[bold magenta]haimllama-cli[/bold magenta]",
            border_style="magenta",
        )
    )

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye.[/dim]")
            break

        if not user_input:
            continue

        # ── Built-in commands ────────────────────────────────────────────
        if user_input.startswith("/"):
            parts = user_input.split(maxsplit=1)
            cmd = parts[0].lower()
            arg = parts[1] if len(parts) > 1 else ""

            if cmd == "/exit":
                console.print("[dim]Goodbye.[/dim]")
                break

            elif cmd == "/help":
                rprint(HELP_TEXT)

            elif cmd == "/clear":
                agent.clear()
                console.print("[dim]Conversation cleared.[/dim]")

            elif cmd == "/tools":
                for schema in TOOL_SCHEMAS:
                    fn = schema["function"]
                    console.print(f"  [cyan]{fn['name']}[/cyan]  {fn['description'][:80]}")

            elif cmd == "/models":
                try:
                    models = client.list_models()
                    for m in models:
                        marker = "[green]●[/green]" if m == model_holder[0] else " "
                        console.print(f"  {marker} {m}")
                except Exception as e:
                    console.print(f"[red]Error:[/red] {e}")

            elif cmd == "/model":
                if not arg:
                    console.print(f"Current model: [cyan]{model_holder[0]}[/cyan]")
                else:
                    model_holder[0] = arg
                    agent.model = arg
                    console.print(f"Switched to [cyan]{arg}[/cyan]")

            elif cmd == "/history":
                for i, msg in enumerate(agent.messages):
                    role = msg["role"].upper()
                    content = str(msg.get("content", ""))[:200]
                    console.print(f"[dim]{i}[/dim] [bold]{role}[/bold]: {content}")

            elif cmd == "/save":
                path = arg or "conversation.json"
                try:
                    Path(path).write_text(json.dumps(agent.messages, indent=2))
                    console.print(f"Saved to [cyan]{path}[/cyan]")
                except Exception as e:
                    console.print(f"[red]Error:[/red] {e}")

            elif cmd == "/load":
                if not arg:
                    console.print("[red]Usage:[/red] /load <file>")
                else:
                    try:
                        agent.messages = json.loads(Path(arg).read_text())
                        console.print(f"Loaded {len(agent.messages)} messages from [cyan]{arg}[/cyan]")
                    except Exception as e:
                        console.print(f"[red]Error:[/red] {e}")

            else:
                console.print(f"[red]Unknown command:[/red] {cmd}. Type /help.")

            continue

        # ── Run agent ────────────────────────────────────────────────────
        streaming_active = False
        try:
            result = agent.run(user_input)
            finish_stream()
            # If the model didn't stream (pure tool workflow), print final answer
            if result and not streaming_active:
                print_assistant(result)
        except KeyboardInterrupt:
            finish_stream()
            console.print("\n[yellow]Interrupted.[/yellow]")
        except Exception as e:
            finish_stream()
            console.print(f"\n[red]Error:[/red] {e}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="haimllama-cli — a local AI agent powered by Ollama",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
            Examples:
              haimllama-cli
              haimllama-cli -m llama3.2
              haimllama-cli "list files in the current directory"
              echo "summarise this text" | haimllama-cli
        """),
    )
    parser.add_argument("query", nargs="?", help="One-shot query (non-interactive)")
    parser.add_argument("-m", "--model", default=None, help="Ollama model to use")
    parser.add_argument("--url", default=OLLAMA_URL, help=f"Ollama base URL (default: {OLLAMA_URL})")
    parser.add_argument("--no-tools", action="store_true", help="Disable all tools")
    parser.add_argument("--max-iter", type=int, default=20, help="Max agentic iterations (default 20)")
    parser.add_argument("--list-models", action="store_true", help="List available models and exit")
    args = parser.parse_args()

    cfg = load_config()
    model = args.model or cfg.get("default_model", DEFAULT_MODEL)

    client = OllamaClient(args.url)

    # Verify Ollama is reachable
    try:
        available = client.list_models()
    except ConnectionError as e:
        console.print(f"[red]{e}[/red]")
        sys.exit(1)

    if args.list_models:
        for m in available:
            marker = "[green]●[/green]" if m == model else " "
            console.print(f"  {marker} {m}")
        return

    if model not in available:
        console.print(
            f"[yellow]Warning:[/yellow] model [cyan]{model}[/cyan] not found locally. "
            f"Ollama may try to pull it.\n"
            f"Available: {', '.join(available)}"
        )

    system = SYSTEM_PROMPT.format(cwd=os.getcwd())
    agent = Agent(
        model=model,
        client=client,
        system_prompt=system,
        max_iterations=args.max_iter,
    )

    if args.no_tools:
        from tools import TOOL_SCHEMAS as _schemas
        _schemas.clear()

    # ── One-shot mode (query arg or piped stdin) ─────────────────────────
    piped = not sys.stdin.isatty()
    if args.query or piped:
        if piped:
            stdin_text = sys.stdin.read().strip()
            query = f"{args.query}\n\n{stdin_text}" if args.query else stdin_text
        else:
            query = args.query

        tokens_printed = []

        def on_token(tok):
            console.print(tok, end="", markup=False)
            tokens_printed.append(tok)

        agent.on_token = on_token
        agent.on_tool_call = lambda n, a: print_tool_call(n, a)
        agent.on_tool_result = lambda n, r: print_tool_result(n, r)

        result = agent.run(query)
        if not tokens_printed:
            print_assistant(result)
        else:
            console.print()
        return

    # ── Interactive REPL ─────────────────────────────────────────────────
    model_holder = [model]
    repl(agent, client, model_holder)

    # Save last used model
    cfg["default_model"] = model_holder[0]
    save_config(cfg)


if __name__ == "__main__":
    main()
