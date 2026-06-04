"""结构化日志：文件轮转 + 控制台输出，线程安全。

通过环境变量 TIRED_LOG_LEVEL 控制级别（DEBUG/INFO/WARNING/ERROR），默认 INFO。
日志文件写入 logs/tired_detect.log，单文件最大 5 MB，保留最近 7 个备份。
"""

import logging
import logging.handlers
import os
from pathlib import Path


_LOG_REGISTRY: dict[str, logging.Logger] = {}


def get_logger(name: str = "tired_detect") -> logging.Logger:
    """返回已配置的 logger 实例（同一 name 只初始化一次，避免重复 handler）。"""
    if name in _LOG_REGISTRY:
        return _LOG_REGISTRY[name]

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # 由 handler 各自控制等级

    # 避免重复添加（例如模块重载场景）
    if logger.handlers:
        _LOG_REGISTRY[name] = logger
        return logger

    # 文件 handler：写入所有级别，轮转
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    fh = logging.handlers.RotatingFileHandler(
        log_dir / "tired_detect.log",
        maxBytes=5 * 1024 * 1024,  # 5 MB
        backupCount=7,
        encoding="utf-8",
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(levelname)-7s] %(name)s:%(lineno)d  %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(fh)

    # 控制台 handler：按环境变量控制级别
    level_name = os.environ.get("TIRED_LOG_LEVEL", "INFO").upper()
    console_level = getattr(logging, level_name, logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(console_level)
    ch.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(levelname)-7s] %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    logger.addHandler(ch)

    # 抑制第三方库的冗余日志
    for lib in ("PIL", "urllib3", "requests", "mediapipe"):
        logging.getLogger(lib).setLevel(logging.WARNING)

    _LOG_REGISTRY[name] = logger
    return logger


def shutdown_logging() -> None:
    """刷新并关闭所有日志 handler。"""
    for logger in _LOG_REGISTRY.values():
        for handler in list(logger.handlers):
            handler.flush()
            handler.close()
            logger.removeHandler(handler)
    _LOG_REGISTRY.clear()
