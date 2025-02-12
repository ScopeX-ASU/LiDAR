import logging
import logging.handlers
import random
import time
from pathlib import Path


class TimerCtx:
    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start


class CustomFormatter(logging.Formatter):
    """Logging Formatter to add colors and count warning / errors"""

    grey = "\x1b[38;21m"
    yellow = "\x1b[33;21m"
    red = "\x1b[31;21m"
    bold_red = "\x1b[31;1m"
    green = "\x1b[32;21m"
    reset = "\x1b[0m"
    # format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"
    format = "%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: grey + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def setup_default_logging(
    default_level=logging.INFO, default_file_level=logging.INFO, log_path=""
):
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(CustomFormatter())
    logging.root.addHandler(console_handler)
    logging.root.setLevel(default_level)
    if log_path:
        file_handler = logging.handlers.RotatingFileHandler(
            log_path, maxBytes=(1024**2 * 2), backupCount=3
        )
        file_formatter = logging.Formatter(
            "%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s"
        )
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(default_file_level)
        logging.root.addHandler(file_handler)


def get_logger(
    name="default",
    default_level=logging.INFO,
    default_file_level=logging.INFO,
    log_path="",
):
    setup_default_logging(
        default_level=default_level,
        default_file_level=default_file_level,
        log_path=log_path,
    )
    return logging.getLogger(name)


logger = get_logger()


def set_torch_deterministic(random_state: int = 0) -> None:
    random_state = int(random_state) % (2**32)
    random.seed(random_state)


def ensure_dir(dirname, exist_ok: bool = True):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=exist_ok)
