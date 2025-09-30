"""Email assistant package."""

from .logging_config import setup_logging as _setup_logging

version = "0.1.0"

_LOG_PATH = _setup_logging()

__all__ = ["_LOG_PATH", "version"]
