import logging
import sys


def setup_logging(level: int = logging.INFO) -> None:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        fmt='[%(asctime)s] %(levelname)s %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    handler.setFormatter(formatter)
    root = logging.getLogger()
    root.setLevel(level)
    # Avoid duplicate handlers in REPL-like runs
    if not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        root.addHandler(handler) 