"""
logger.py — Centralized logging for the Kira AI Assistant.

Logs API calls, errors, fallbacks, and gestures to logs/kira.log
with timestamps and severity levels.
"""

import logging
import os

from config import LOG_FILE


def setup_logging():
    """Configure the root logger to write to both console and the configured log file."""
    log_dir = os.path.dirname(LOG_FILE)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    root_logger = logging.getLogger()
    # Avoid duplicate handlers if called more than once
    if root_logger.handlers:
        return root_logger

    root_logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # File handler — logs everything
    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    # Console handler — INFO and above
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    return root_logger


def get_logger(name: str) -> logging.Logger:
    """Return a named child logger."""
    return logging.getLogger(name)
