"""Logging setup for haimllama-cli.

Usage
-----
In main.py (once, at startup):
    from logger import setup_logging
    setup_logging(log_file=args.log_file, level=args.log_level, console=args.log_console)

In any other module:
    from logger import get_logger
    log = get_logger(__name__)
    log.info("something happened")
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

DEFAULT_LOG_DIR = Path.home() / ".local" / "share" / "haimllama-cli" / "logs"
_LOG_FORMAT = "%(asctime)s [%(levelname)-8s] %(name)s — %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Module-level flag so re-entrant calls are safe
_initialized = False


def setup_logging(
    log_file: str | None = None,
    level: str = "INFO",
    console: bool = False,
) -> logging.Logger:
    """Configure the haimllama root logger.

    Args:
        log_file:  Explicit path for the log file.  When *None* a date-stamped
                   file is created under DEFAULT_LOG_DIR.
        level:     One of DEBUG / INFO / WARNING / ERROR (case-insensitive).
        console:   If True, also emit log records to stderr.

    Returns the configured ``haimllama`` root logger.
    """
    global _initialized

    numeric_level = getattr(logging, level.upper(), logging.INFO)

    root = logging.getLogger("haimllama")
    root.setLevel(numeric_level)
    # Clear any handlers from a previous call (e.g. in tests)
    root.handlers.clear()

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    # ── File handler ────────────────────────────────────────────────────────
    if log_file is None:
        DEFAULT_LOG_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d")
        log_file = str(DEFAULT_LOG_DIR / f"haimllama-{stamp}.log")

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(formatter)
    root.addHandler(fh)

    # ── Optional console handler ─────────────────────────────────────────────
    if console:
        ch = logging.StreamHandler(sys.stderr)
        ch.setFormatter(formatter)
        root.addHandler(ch)

    _initialized = True
    root.debug("Logging initialised — file=%s  level=%s", log_file, level.upper())
    return root


def get_logger(name: str) -> logging.Logger:
    """Return a child logger in the *haimllama* namespace.

    If :func:`setup_logging` has never been called (e.g. during testing) a
    NullHandler is attached so no "No handlers could be found" warnings appear.
    """
    logger = logging.getLogger(f"haimllama.{name}")
    if not _initialized and not logging.getLogger("haimllama").handlers:
        logging.getLogger("haimllama").addHandler(logging.NullHandler())
    return logger
