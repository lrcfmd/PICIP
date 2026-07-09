import logging
import os


def newLogger(name: str, level: str | None = None):
    """Create a standard library logger.

    Args:
        name: Logger name (typically __name__)
        level: Log level (DEBUG, INFO, WARNING, ERROR). Defaults to LOG_LEVEL env var or INFO.

    Returns:
        Standard library logger instance
    """
    if level is None:
        level = os.environ.get("LOG_LEVEL", "INFO").upper()

    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logger = logging.getLogger(name)
    logger.setLevel(numeric_level)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(numeric_level)
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger
