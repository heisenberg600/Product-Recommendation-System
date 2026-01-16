"""Logging configuration using loguru"""

import sys
from loguru import logger

from app.core.config import settings


def setup_logging() -> None:
    """Configure application logging"""

    # Remove default handler
    logger.remove()

    # Console handler with color
    log_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )

    logger.add(
        sys.stderr,
        format=log_format,
        level="DEBUG" if settings.debug else "INFO",
        colorize=True,
    )

    # File handler for production
    if settings.environment == "production":
        logger.add(
            "logs/app.log",
            rotation="10 MB",
            retention="7 days",
            compression="zip",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
            level="INFO",
        )


# Export logger instance
__all__ = ["logger", "setup_logging"]
