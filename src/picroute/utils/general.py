"""Small runtime utilities shared by the router entry points."""

import logging
import logging.handlers
import random
import time
from pathlib import Path

_LOG_FORMAT = "%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s: %(message)s"


class TimerCtx:
    """Measure a block with a monotonic high-resolution clock."""

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *_args):
        self.interval = time.perf_counter() - self.start


def configure_logging(
    level: int = logging.INFO,
    *,
    log_path: str | Path | None = None,
) -> logging.Logger:
    logger = logging.getLogger("picroute")
    logger.setLevel(level)
    logger.propagate = False
    if not logger.handlers:
        console = logging.StreamHandler()
        console.setFormatter(logging.Formatter(_LOG_FORMAT))
        logger.addHandler(console)
    if log_path is not None:
        destination = Path(log_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not any(
            isinstance(handler, logging.FileHandler)
            and Path(handler.baseFilename) == destination.resolve()
            for handler in logger.handlers
        ):
            file_handler = logging.handlers.RotatingFileHandler(
                destination, maxBytes=2 * 1024**2, backupCount=3
            )
            file_handler.setFormatter(logging.Formatter(_LOG_FORMAT))
            logger.addHandler(file_handler)
    return logger


logger = configure_logging()


def seed_random(random_state: int = 0) -> None:
    random.seed(int(random_state) % 2**32)


def ensure_directory(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory
