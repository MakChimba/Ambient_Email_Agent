"""Central logging setup for the email assistant."""

from __future__ import annotations

import logging
import os
from pathlib import Path
__all__ = ["setup_logging"]

_LOG_PATH_ENV = "EMAIL_ASSISTANT_LOG_PATH"
_LOG_LEVEL_ENV = "EMAIL_ASSISTANT_LOG_LEVEL"
_DEFAULT_LOG_PATH = "logs/email_assistant.log"


def _has_handler(root: logging.Logger, candidate: Path) -> bool:
    return any(
        isinstance(handler, logging.FileHandler)
        and Path(getattr(handler, "baseFilename", "")).resolve() == candidate
        for handler in root.handlers
    )


def _ensure_root_level(root: logging.Logger, level: int) -> None:
    if root.level > level or root.level == logging.NOTSET:
        root.setLevel(level)


def _fallback_basic_config(level: int) -> None:
    if not logging.getLogger().handlers:
        logging.basicConfig(level=level)


def _to_level(value: str | None) -> int:
    if not value:
        return logging.INFO
    try:
        return logging.getLevelName(value.upper())  # type: ignore[arg-type]
    except Exception:
        return logging.INFO


def setup_logging() -> Path:
    """Attach a shared file handler capturing frontend/backend logs.

    Returns the resolved log path so callers can surface it in UIs/tests.
    """

    log_level = _to_level(os.getenv(_LOG_LEVEL_ENV))
    log_path = Path(os.getenv(_LOG_PATH_ENV, _DEFAULT_LOG_PATH)).expanduser().resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    _fallback_basic_config(log_level)
    _ensure_root_level(root, log_level)

    if not _has_handler(root, log_path):
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(log_level)
        formatter = logging.Formatter(
            "%(asctime)s %(levelname)s [%(name)s] %(message)s"
        )
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    return log_path
