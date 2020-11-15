import logging


class CustomLogger(logging.Logger):
    pass


def setup_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel('INFO')
    # NOTE if print log twice, please refer to https://bit.ly/2NuC9qV
    logger.propagate = False

    if logger.handlers:
        logger.handlers[0].close()
        logger.handlers = []
    handle = logging.StreamHandler()
    formatter = logging.Formatter('%(levelname).1s ' +
                                  '%(asctime)s - %(filename)s:%(lineno)d]' +
                                  ' %(message)s')
    handle.setFormatter(formatter)

    logger.addHandler(handle)

    return logger
