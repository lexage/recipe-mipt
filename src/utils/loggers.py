import os
import logging
import distutils.dir_util


def create_logging(log_filename=None, level=logging.INFO):
    """Configure global logging with optional file output"""

    if log_filename is not None:

        if os.path.isfile(log_filename):
            os.remove(log_filename)
        distutils.dir_util.mkpath(os.path.dirname(log_filename))

        logging.basicConfig(
            level=level,
            format="%(message)s",
            handlers=[
                logging.FileHandler(log_filename),
                logging.StreamHandler()
            ]
        )

    else:
        logging.basicConfig(
            level=level,
            format="%(message)s",
            handlers=[
                logging.StreamHandler()
            ]
        )
