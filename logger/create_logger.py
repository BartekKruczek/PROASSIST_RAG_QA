import logging
import os


def create_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Create a logger with the specified name and logging level.

    Args:
        name (str): The name of the logger.
        level (int): The logging level (default is logging.INFO).

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.hasHandlers():
        base_dir = os.path.dirname(os.path.abspath(__file__))
        log_path = os.path.join(base_dir, "logfile.log")

        fh = logging.FileHandler(log_path)
        fh.setLevel(level)
        fh.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )

        ch = logging.StreamHandler()
        ch.setLevel(level)
        ch.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )

        logger.addHandler(fh)
        logger.addHandler(ch)

    return logger
