"""Module to hold the lowest level objects used in the CardShark engine

Currently only NamedObject which represents unique (named) objects which we
may want to identify by name, for reasons of debugging, or displaying info to
users etc.
"""

# python stuff
from collections import defaultdict

# cardshark stuff
from cardshark.logging import Logger


class NamedObject:
    """Base class for unique named objects in the project

    Can also hold a :py:class:`Logger` that allows for individual objects to log events

    Args:
        logger (Logger): The logger that should be used by this object.
        name (str): The name of this object instance. Default will be
        "<name of derived class>_<N instances of derived class>".
    """

    instance_counts = defaultdict(int)

    def __init__(self, logger: Logger = None, name: str = ""):
        ## set optional properties
        if name == "":
            self.name = (
                self.__class__.__name__
                + "_"
                + str(NamedObject.instance_counts[self.__class__.__name__])
            )
        else:
            self.name = name

        self.logger = logger

        NamedObject.instance_counts[self.__class__.__name__] += 1

    ## wrap the logger functions.
    def error(self, *messages):
        """log an error message to the logger associated with this object (if it exists)"""
        if self.logger is not None:
            self.logger.error("{" + self.name + "}", *messages)

    def warn(self, *messages):
        """log a warning message to the logger associated with this object (if it exists)"""
        if self.logger is not None:
            self.logger.warn("{" + self.name + "}", *messages)

    def info(self, *messages):
        """log an info message to the logger associated with this object (if it exists)"""
        if self.logger is not None:
            self.logger.info("{" + self.name + "}", *messages)

    def debug(self, *messages):
        """log a debug message to the logger associated with this object (if it exists)"""
        if self.logger is not None:
            self.logger.debug("{" + self.name + "}", *messages)

    def trace(self, *messages):
        """log a trace message to the logger associated with this object (if it exists)"""
        if self.logger is not None:
            self.logger.trace("{" + self.name + "}", *messages)
