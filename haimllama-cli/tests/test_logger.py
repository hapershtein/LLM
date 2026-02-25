"""Unit tests for logger.py — setup_logging and get_logger."""

import logging
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import logger as logger_module
from logger import setup_logging, get_logger


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def reset_logger():
    """Reset the haimllama logger and module flag before each test."""
    root = logging.getLogger("haimllama")
    root.handlers.clear()
    logger_module._initialized = False
    yield
    root.handlers.clear()
    logger_module._initialized = False


# ── setup_logging ──────────────────────────────────────────────────────────────

class TestSetupLogging:
    def test_creates_log_file(self, tmp_path):
        log_file = str(tmp_path / "test.log")
        setup_logging(log_file=log_file)
        assert Path(log_file).exists()

    def test_writes_to_log_file(self, tmp_path):
        log_file = str(tmp_path / "test.log")
        setup_logging(log_file=log_file, level="DEBUG")
        log = get_logger("test_write")
        log.info("hello from test")
        # Flush handlers
        for h in logging.getLogger("haimllama").handlers:
            h.flush()
        content = Path(log_file).read_text()
        assert "hello from test" in content

    def test_respects_info_level(self, tmp_path):
        log_file = str(tmp_path / "level.log")
        setup_logging(log_file=log_file, level="INFO")
        log = get_logger("test_level")
        log.debug("should not appear")
        log.info("should appear")
        for h in logging.getLogger("haimllama").handlers:
            h.flush()
        content = Path(log_file).read_text()
        assert "should appear" in content
        assert "should not appear" not in content

    def test_respects_debug_level(self, tmp_path):
        log_file = str(tmp_path / "debug.log")
        setup_logging(log_file=log_file, level="DEBUG")
        log = get_logger("test_debug")
        log.debug("debug message")
        for h in logging.getLogger("haimllama").handlers:
            h.flush()
        content = Path(log_file).read_text()
        assert "debug message" in content

    def test_console_handler_added(self, tmp_path, capsys):
        log_file = str(tmp_path / "console.log")
        setup_logging(log_file=log_file, level="INFO", console=True)
        root = logging.getLogger("haimllama")
        handler_types = [type(h).__name__ for h in root.handlers]
        assert "StreamHandler" in handler_types
        assert "FileHandler" in handler_types

    def test_no_console_handler_by_default(self, tmp_path):
        log_file = str(tmp_path / "no_console.log")
        setup_logging(log_file=log_file, console=False)
        root = logging.getLogger("haimllama")
        stream_handlers = [h for h in root.handlers if type(h).__name__ == "StreamHandler"]
        assert stream_handlers == []

    def test_sets_initialized_flag(self, tmp_path):
        log_file = str(tmp_path / "flag.log")
        assert not logger_module._initialized
        setup_logging(log_file=log_file)
        assert logger_module._initialized

    def test_reinitialize_clears_old_handlers(self, tmp_path):
        log_file1 = str(tmp_path / "a.log")
        log_file2 = str(tmp_path / "b.log")
        setup_logging(log_file=log_file1)
        setup_logging(log_file=log_file2)
        root = logging.getLogger("haimllama")
        # Should have exactly one handler after re-init (not two)
        assert len(root.handlers) == 1

    def test_returns_root_logger(self, tmp_path):
        log_file = str(tmp_path / "ret.log")
        returned = setup_logging(log_file=log_file)
        assert returned is logging.getLogger("haimllama")

    def test_log_format_contains_level_and_message(self, tmp_path):
        log_file = str(tmp_path / "fmt.log")
        setup_logging(log_file=log_file)
        log = get_logger("fmt_test")
        log.warning("format check")
        for h in logging.getLogger("haimllama").handlers:
            h.flush()
        content = Path(log_file).read_text()
        assert "WARNING" in content
        assert "format check" in content


# ── get_logger ─────────────────────────────────────────────────────────────────

class TestGetLogger:
    def test_returns_logger_with_haimllama_prefix(self):
        log = get_logger("mymodule")
        assert log.name == "haimllama.mymodule"

    def test_child_logger_propagates_to_root(self, tmp_path):
        log_file = str(tmp_path / "propagate.log")
        setup_logging(log_file=log_file)
        child = get_logger("child")
        child.info("propagated message")
        for h in logging.getLogger("haimllama").handlers:
            h.flush()
        content = Path(log_file).read_text()
        assert "propagated message" in content

    def test_no_error_when_not_initialized(self):
        # get_logger before setup_logging should not raise
        log = get_logger("early")
        log.info("early message")  # Should be silently swallowed by NullHandler

    def test_null_handler_attached_when_not_initialized(self):
        get_logger("null_test")
        root = logging.getLogger("haimllama")
        handler_types = [type(h).__name__ for h in root.handlers]
        assert "NullHandler" in handler_types
