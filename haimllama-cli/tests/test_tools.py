"""Unit tests for tools.py — all 8 tools, dispatch, and helpers."""

import sys
import os
import tempfile
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Allow importing from parent directory
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools import (
    run_shell,
    read_file,
    write_file,
    list_dir,
    find_files,
    grep,
    python_eval,
    fetch_url,
    dispatch,
    _coerce_types,
    TOOL_SCHEMAS,
    TOOL_MAP,
)


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp(tmp_path):
    """Provide a temp directory with a few preset files."""
    (tmp_path / "hello.txt").write_text("hello world\nline two\nline three\n")
    (tmp_path / "data.json").write_text('{"key": "value"}')
    (tmp_path / "script.py").write_text("def foo():\n    return 42\n")
    sub = tmp_path / "subdir"
    sub.mkdir()
    (sub / "nested.txt").write_text("nested content")
    return tmp_path


# ── run_shell ──────────────────────────────────────────────────────────────────

class TestRunShell:
    def test_basic_echo(self):
        result = run_shell("echo hello")
        assert "hello" in result

    def test_captures_stdout(self):
        result = run_shell("echo stdout_content")
        assert "stdout_content" in result

    def test_captures_stderr(self):
        result = run_shell("echo err >&2")
        assert "err" in result
        assert "[stderr]" in result

    def test_nonzero_exit_code_reported(self):
        # `exit 42` produces no stdout, so the no-output branch fires
        result = run_shell("exit 42", timeout=5)
        assert "42" in result  # exit code appears in either branch

    def test_timeout_returns_error(self):
        result = run_shell("sleep 10", timeout=1)
        assert "[error]" in result
        assert "timed out" in result

    def test_cwd_respected(self, tmp_path):
        result = run_shell("pwd", cwd=str(tmp_path))
        assert str(tmp_path) in result or tmp_path.name in result

    def test_empty_output(self):
        result = run_shell("true")
        assert "exit code 0" in result or result == "(exit code 0, no output)"

    def test_invalid_command_returns_error(self):
        result = run_shell("this_command_does_not_exist_xyz")
        # Should return error (stderr or exit code)
        assert result  # something returned


# ── read_file ──────────────────────────────────────────────────────────────────

class TestReadFile:
    def test_reads_full_file(self, tmp):
        result = read_file(str(tmp / "hello.txt"))
        assert "hello world" in result
        assert "line two" in result
        assert "line three" in result

    def test_line_numbers_in_output(self, tmp):
        result = read_file(str(tmp / "hello.txt"))
        assert "1:" in result or "   1:" in result

    def test_start_line(self, tmp):
        result = read_file(str(tmp / "hello.txt"), start_line=2)
        assert "line two" in result
        assert "hello world" not in result

    def test_end_line(self, tmp):
        result = read_file(str(tmp / "hello.txt"), end_line=1)
        assert "hello world" in result
        assert "line two" not in result

    def test_line_range(self, tmp):
        result = read_file(str(tmp / "hello.txt"), start_line=2, end_line=2)
        assert "line two" in result
        assert "hello world" not in result
        assert "line three" not in result

    def test_missing_file_returns_error(self):
        result = read_file("/nonexistent/path/file.txt")
        assert "[error]" in result

    def test_empty_file(self, tmp):
        (tmp / "empty.txt").write_text("")
        result = read_file(str(tmp / "empty.txt"))
        assert "empty" in result

    def test_large_file_refused(self, tmp):
        big = tmp / "big.bin"
        big.write_bytes(b"x" * (2_000_001))
        result = read_file(str(big))
        assert "[error]" in result
        assert "2MB" in result


# ── write_file ─────────────────────────────────────────────────────────────────

class TestWriteFile:
    def test_creates_new_file(self, tmp):
        path = str(tmp / "new.txt")
        result = write_file(path, "hello")
        assert "Written" in result
        assert Path(path).read_text() == "hello"

    def test_overwrites_existing(self, tmp):
        path = str(tmp / "hello.txt")
        write_file(path, "new content")
        assert Path(path).read_text() == "new content"

    def test_append_mode(self, tmp):
        path = str(tmp / "hello.txt")
        original = Path(path).read_text()
        write_file(path, " appended", append=True)
        assert Path(path).read_text() == original + " appended"

    def test_creates_parent_dirs(self, tmp):
        path = str(tmp / "a" / "b" / "c" / "file.txt")
        result = write_file(path, "deep")
        assert "Written" in result
        assert Path(path).read_text() == "deep"

    def test_reports_char_count(self, tmp):
        path = str(tmp / "counted.txt")
        result = write_file(path, "12345")
        assert "5" in result


# ── list_dir ───────────────────────────────────────────────────────────────────

class TestListDir:
    def test_shows_files(self, tmp):
        result = list_dir(str(tmp))
        assert "hello.txt" in result
        assert "data.json" in result

    def test_shows_subdirectory(self, tmp):
        result = list_dir(str(tmp))
        assert "subdir" in result
        assert "[DIR]" in result

    def test_hides_dotfiles_by_default(self, tmp):
        (tmp / ".hidden").write_text("secret")
        result = list_dir(str(tmp))
        assert ".hidden" not in result

    def test_shows_dotfiles_when_requested(self, tmp):
        (tmp / ".hidden").write_text("secret")
        result = list_dir(str(tmp), show_hidden=True)
        assert ".hidden" in result

    def test_nonexistent_path_returns_error(self):
        result = list_dir("/nonexistent/path/xyz")
        assert "[error]" in result

    def test_empty_directory(self, tmp):
        empty = tmp / "empty_dir"
        empty.mkdir()
        result = list_dir(str(empty))
        assert "empty" in result

    def test_file_sizes_shown(self, tmp):
        result = list_dir(str(tmp))
        assert "B" in result or "KB" in result


# ── find_files ─────────────────────────────────────────────────────────────────

class TestFindFiles:
    def test_finds_txt_files(self, tmp):
        result = find_files("*.txt", root=str(tmp))
        assert "hello.txt" in result

    def test_recursive_glob(self, tmp):
        result = find_files("**/*.txt", root=str(tmp))
        assert "nested.txt" in result

    def test_no_match_returns_message(self, tmp):
        result = find_files("*.rs", root=str(tmp))
        assert "No files matched" in result or result == f"No files matched '*.rs' under {str(tmp)}"

    def test_finds_json_files(self, tmp):
        result = find_files("*.json", root=str(tmp))
        assert "data.json" in result

    def test_finds_py_files(self, tmp):
        result = find_files("*.py", root=str(tmp))
        assert "script.py" in result


# ── grep ───────────────────────────────────────────────────────────────────────

class TestGrep:
    def test_finds_pattern_in_file(self, tmp):
        result = grep("hello", path=str(tmp / "hello.txt"))
        assert "hello world" in result

    def test_no_match_returns_message(self, tmp):
        result = grep("zzznomatch", path=str(tmp / "hello.txt"))
        assert "No matches" in result

    def test_case_insensitive(self, tmp):
        result = grep("HELLO", path=str(tmp / "hello.txt"), case_insensitive=True)
        assert "hello world" in result

    def test_case_sensitive_miss(self, tmp):
        result = grep("HELLO", path=str(tmp / "hello.txt"), case_insensitive=False)
        assert "No matches" in result

    def test_context_lines(self, tmp):
        result = grep("line two", path=str(tmp / "hello.txt"), context_lines=1)
        assert "hello world" in result   # line before
        assert "line three" in result    # line after

    def test_glob_filter(self, tmp):
        result = grep("hello", path=str(tmp), glob="*.txt")
        assert "hello world" in result

    def test_glob_filter_excludes_non_matching(self, tmp):
        result = grep("key", path=str(tmp), glob="*.txt")
        # "key" is only in data.json, not in .txt files
        assert "No matches" in result

    def test_invalid_regex_returns_error(self, tmp):
        result = grep("[invalid", path=str(tmp / "hello.txt"))
        assert "[error]" in result

    def test_searches_directory_recursively(self, tmp):
        result = grep("nested", path=str(tmp))
        assert "nested content" in result

    def test_line_numbers_in_output(self, tmp):
        result = grep("hello", path=str(tmp / "hello.txt"))
        assert ":1" in result  # line 1


# ── python_eval ────────────────────────────────────────────────────────────────

class TestPythonEval:
    def test_basic_print(self):
        result = python_eval("print('hello')")
        assert result == "hello"

    def test_arithmetic(self):
        result = python_eval("print(2 + 2)")
        assert result == "4"

    def test_multiline_code(self):
        code = "x = 10\ny = 20\nprint(x + y)"
        result = python_eval(code)
        assert result == "30"

    def test_no_output(self):
        result = python_eval("x = 42")
        assert result == "(no output)"

    def test_syntax_error_returns_traceback(self):
        result = python_eval("def broken(:")
        assert "Error" in result or "Traceback" in result or "SyntaxError" in result

    def test_runtime_error_returns_traceback(self):
        result = python_eval("raise ValueError('oops')")
        assert "ValueError" in result
        assert "oops" in result

    def test_imports_work(self):
        result = python_eval("import math\nprint(math.pi > 3)")
        assert "True" in result

    def test_indented_code_dedented(self):
        # Common case: model sends indented code
        code = "    print('dedented')"
        result = python_eval(code)
        assert "dedented" in result

    def test_stderr_captured(self):
        result = python_eval("import sys\nprint('err', file=sys.stderr)")
        assert "err" in result


# ── fetch_url ──────────────────────────────────────────────────────────────────

class TestFetchUrl:
    def _make_mock_client(self, resp):
        """Build a context-manager-compatible mock httpx.Client."""
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = resp
        return mock_client

    def test_successful_get(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"content-type": "text/html"}
        mock_resp.text = "<html>hello</html>"

        with patch("tools.httpx.Client", return_value=self._make_mock_client(mock_resp)):
            result = fetch_url("http://example.com")

        assert "[HTTP 200]" in result
        assert "hello" in result

    def test_truncates_long_response(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"content-type": "text/plain"}
        mock_resp.text = "x" * 25_000

        with patch("tools.httpx.Client", return_value=self._make_mock_client(mock_resp)):
            result = fetch_url("http://example.com")

        assert "truncated" in result
        assert len(result) < 25_000

    def test_connection_error_returns_error(self):
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(side_effect=Exception("connection refused"))
        mock_client.__exit__ = MagicMock(return_value=False)
        with patch("tools.httpx.Client", return_value=mock_client):
            result = fetch_url("http://unreachable.invalid")
        assert "[error]" in result


# ── _coerce_types ──────────────────────────────────────────────────────────────

class TestCoerceTypes:
    def test_string_true(self):
        assert _coerce_types({"flag": "true"}) == {"flag": True}

    def test_string_false(self):
        assert _coerce_types({"flag": "false"}) == {"flag": False}

    def test_string_true_uppercase(self):
        assert _coerce_types({"flag": "TRUE"}) == {"flag": True}

    def test_integer_string(self):
        assert _coerce_types({"n": "42"}) == {"n": 42}

    def test_float_string(self):
        result = _coerce_types({"f": "3.14"})
        assert abs(result["f"] - 3.14) < 0.001

    def test_plain_string_unchanged(self):
        assert _coerce_types({"s": "hello"}) == {"s": "hello"}

    def test_already_bool_unchanged(self):
        assert _coerce_types({"b": True}) == {"b": True}

    def test_already_int_unchanged(self):
        assert _coerce_types({"n": 7}) == {"n": 7}

    def test_empty_dict(self):
        assert _coerce_types({}) == {}

    def test_mixed_types(self):
        result = _coerce_types({"a": "true", "b": "5", "c": "hello", "d": 99})
        assert result == {"a": True, "b": 5, "c": "hello", "d": 99}


# ── dispatch ───────────────────────────────────────────────────────────────────

class TestDispatch:
    def test_known_tool(self):
        result = dispatch("shell", {"command": "echo hi"})
        assert "hi" in result

    def test_unknown_tool(self):
        result = dispatch("nonexistent_tool", {})
        assert "[error]" in result
        assert "Unknown tool" in result

    def test_bad_args_returns_error(self):
        result = dispatch("shell", {"totally_wrong_arg": "x"})
        assert "[error]" in result

    def test_coerces_bool_strings(self):
        with tempfile.TemporaryDirectory() as d:
            # show_hidden as string should be coerced to bool
            result = dispatch("list_dir", {"path": d, "show_hidden": "false"})
            assert "[error]" not in result

    def test_all_tools_registered(self):
        expected = {"shell", "read_file", "write_file", "list_dir",
                    "find_files", "grep", "python_eval", "fetch_url"}
        assert set(TOOL_MAP.keys()) == expected


# ── TOOL_SCHEMAS ───────────────────────────────────────────────────────────────

class TestToolSchemas:
    def test_all_schemas_have_required_fields(self):
        for schema in TOOL_SCHEMAS:
            assert schema["type"] == "function"
            fn = schema["function"]
            assert "name" in fn
            assert "description" in fn
            assert "parameters" in fn

    def test_schema_count_matches_tool_map(self):
        schema_names = {s["function"]["name"] for s in TOOL_SCHEMAS}
        assert schema_names == set(TOOL_MAP.keys())
