import logging
import typing


def get_logger(
    log_level: int = logging.INFO,
    name: typing.Optional[str] = None
) -> logging.Logger:
    if not name:
        name = __name__
    
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    formatter = logging.Formatter("%(name)s %(asctime)s %(levelname)s %(message)s")
    handler_stream = logging.StreamHandler()

    handler_stream.setFormatter(formatter)

    logger.addHandler(handler_stream)

    return logger