"""Logger object and log decorator."""

import sys
import logging
import colorlog
from functools import wraps
from typing import Any, Callable, TypeVar


def configure_logger():
    """Configure logger for the standard out (command line) stream and save logs to file"""
    # make log formatters
    stream_formatter = colorlog.ColoredFormatter(
        "%(log_color)s%(asctime)-15s %(levelname).1s "
        "[%(filename)s:%(funcName)s:%(lineno)d] %(message)s",
        "%Y-%m-%d %H:%M:%S",
    )

    file_formatter = logging.Formatter(
        "%(asctime)-15s %(levelname).1s [%(filename)s:%(funcName)s:%(lineno)d]"
        " %(message)s",
        "%Y-%m-%d %H:%M:%S",
    )

    log_file = "Hazen_logger.log"
    stream_handler = logging.StreamHandler(sys.stdout)
    file_handler = logging.FileHandler(log_file)

    # set formatters
    stream_handler.setFormatter(stream_formatter)
    file_handler.setFormatter(file_formatter)

    # add handlers
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)


logger = logging.getLogger(__name__)
configure_logger()

T = TypeVar("T")


def log(fn: Callable[..., T]) -> Callable[..., T]:
    """Log the inputs and outputs to a function."""

    @wraps(fn)
    def inner(*args: Any, **kwargs: Any) -> T:
        logger.debug(
            "Calling %s with arguments: (%s) and keyword arguments: (%s)",
            fn.__name__,
            args,
            kwargs,
        )
        result = fn(*args, **kwargs)
        logger.debug("%s returned %s", fn.__name__, result)
        return result

    return inner
