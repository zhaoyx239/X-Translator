from __future__ import annotations

import logging
import os
from pathlib import Path


def setup_logger() -> logging.Logger:
    """初始化日志。"""
    logger = logging.getLogger("xtranslate")
    if logger.handlers:
        return logger

    app_dir = Path(__file__).resolve().parents[1]
    project_root = app_dir.parent
    log_dir_value = os.getenv("XTRANSLATE_LOG_DIR", str(app_dir / "runtime/logs"))
    log_dir = Path(log_dir_value)
    if not log_dir.is_absolute():
        if str(log_dir).startswith("xtranslate/"):
            log_dir = (project_root / log_dir).resolve()
        else:
            log_dir = (app_dir / log_dir).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "xtranslate.log"

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.propagate = False
    return logger
