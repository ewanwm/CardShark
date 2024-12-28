"""Lightweight module for logging console messages

Main class is the Logger class which can print messages to console or to a file,
each one can have it's own logging level. These can be attached to individual
objects in the engine and allows each of them to log their own messages to different
locations and with different levels of detail.

There is also a global "main_logger" for more general messages, or ones which don't 
make sense to attach to a single object (e.g. to note creation or deletion of loggers,
to log messages in top level functions etc.).

The verbosity levels are defined by the LogLevel enum class.
"""

from enum import Enum
import datetime

class LogLevel(Enum):
    """Possible log levels for Logger objects
    """
    SILENT = 0
    ERROR = 1
    WARNING = 2
    INFO = 3
    DEBUG = 4
    TRACE = 5

## set global default log level
LOG_LEVEL = LogLevel.TRACE


class Logger:
    """
    Used to log events, with a specified level of verbosity.
    Can be used by multiple objects, and optionally log to a file.

    Args:
        log_level (LogLevel):
            Specifies the level that this logger will log at. Will only print messages whose 
            minimum level is above this value.
        name (str):
            The name of this logger, allows to more easily tell where the log message is 
            coming from.
        to_file (bool):
            Whether to log messages to a file or to stdout.
        file_name (std):
            The name of the file to log to if to_file == True. Will default to "logs/<name>.log".
    """

    nLoggers = 0

    def __init__(
        self,
        log_level: LogLevel = LOG_LEVEL,
        name: str = "",
        to_file: bool = False,
        file_name: str = "",
    ):
        self._log_level = log_level
        if name == "":
            self.name = "Logger_" + str(Logger.nLoggers)
        else:
            self.name = name

        self._to_file = to_file
        if to_file:
            if file_name == "":
                self._log_file_name = "logs/" + self.name + ".log"
            else:
                self._log_file_name = file_name

            ## make the file, overwrite if one exists already
            with open(self._log_file_name, "w", encoding="utf-8") as f:
                f.close()

        print(
            datetime.datetime.now(),
            "- Logger",
            self.name,
            "created with log level",
            self._log_level,
        )

        Logger.nLoggers += 1

    def __delete__(self):
        return

    def _format(self, message_type):
        return "[{type}]-[{time}]-[{name}]:".format(
            type=message_type, time=datetime.datetime.now(), name=self.name
        )

    def _log(self, message_type: str, min_level: LogLevel, *messages, indent=0):

        # if log level below min value then dont log nothin'
        if self._log_level.value < min_level.value:
            return

        indent_str = ""
        for _ in range(indent):
            indent_str += " "

        if self._to_file:
            with open(self._log_file_name, "a", encoding="utf-8") as f:
                print(indent_str + self._format(message_type), *messages, file=f)

        else:
            print(indent_str + self._format(message_type), *messages)

    def set_log_level(self, level: LogLevel) -> None:
        """Set the log_level of this Logger
        """

        self._log_level = level

    def get_log_level(self) -> LogLevel:
        """Get the log_level of this Logger
        """

        return self._log_level

    def error(self, *messages):
        """Log an error message to this logger

        Will always show unless log_level is set to SILENT.
        """
        self._log("ERROR", LogLevel.ERROR, *messages)

    def warn(self, *messages):
        """Log a warning to the user to this logger

        Will show if log_level >= WARNING.
        """
        self._log("WARN", LogLevel.WARNING, *messages)

    def info(self, *messages):
        """Log an info message to this logger

        Will show if log_level >= INFO.
        """
        self._log("INFO", LogLevel.INFO, *messages)

    def debug(self, *messages):
        """Log a debug message to this logger

        Will show if log_level >= DEBUG
        """
        self._log("DEBUG", LogLevel.DEBUG, *messages, indent=1)

    def trace(self, *messages):
        """Log a trace message to this logger

        Will show if log_level >= TRACE
        """
        self._log("TRACE", LogLevel.TRACE, *messages, indent=2)


main_logger = Logger(name="main_logger")


def error(*messages):
    """Log an error message to the main logger

    Will always show unless log level is set to SILENT.
    """
    main_logger.error(*messages)


def warn(*messages):
    """Log a warning to the user to the main logger

    Will show if log_level >= WARNING.
    """
    main_logger.warn(*messages)


def info(*messages):
    """Log an info message to the main logger

    Will show if log_level >= INFO.
    """
    main_logger.info(*messages)


def debug(*messages):
    """Log a debug message to the main logger

    Will show if log_level >= DEBUG
    """
    main_logger.debug(*messages)


def trace(*messages):
    """Log a trace message to the main logger

    Will show if log_level >= TRACE
    """
    main_logger.trace(*messages)
