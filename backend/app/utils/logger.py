"""
./backend/app/utils/logger.py

This module contains the logger configuration for the application.
"""

import logging
from pathlib import Path
from backend.app.core.config import settings


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        # prevent multiple handlers in interactive environments
        logger.setLevel(logging.INFO)

        log_dir = Path(settings.LOG_DIR)
        log_dir.mkdir(exist_ok=True, parents=True)

        log_file = log_dir / "app.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)

        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s"
            )
        file_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        
    return logger