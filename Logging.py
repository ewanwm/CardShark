from enum import Enum
import datetime

logLevels = Enum("logLevels", "kSilent kError kWarning kInfo kDebug kTrace")

## set global default log level
LOG_LEVEL = logLevels.kTrace

GAME_LOG_LEVEL = logLevels.kInfo

class Logger:
    """
    Used to log events, with a specified level of verbosity.
    Can be used by multiple objects, and optionally log to a file.

    Args:
        logLevel (logLevels): 
            Specifies the level that this logger will log at. Will only print messages whose minimum level is above this value.
        name (str): 
            The name of this logger, allows to more easily tell where the log message is coming from.
        toFile (bool):
            Whether to log messages to a file or to stdout.
        fileName (std):
            The name of the file to log to if toFile == True. Will default to "logs/<name>.log".
    """
    
    nLoggers = 0

    def __init__(
            self, 
            logLevel: logLevels = LOG_LEVEL, 
            name: str = "", 
            toFile: bool = False, 
            fileName: str = ""
        ):
        
        self.logLevel = logLevel
        if(name == ""):
            self.name = "Logger_" + str(Logger.nLoggers)
        else:
            self.name = name

        self.printToFile = toFile
        if toFile:
            if fileName == "":
                self.printFileName = "logs/" + self.name + ".log"
            else:
                self.printFileName = fileName

            ## make the file, overwrite if one exists already
            f = open(self.printFileName, "w")
            f.close()

        print(datetime.datetime.now(), "- Logger", self.name, "created with log level", self.logLevel)

        Logger.nLoggers += 1

    def __delete__(self):
        return
        
    def format(self, messageType):
        return "[{type}]-[{time}]-[{name}]:".format(type=messageType, time=datetime.datetime.now(), name=self.name)

    def log(self, messageType, minLevel, *messages, indent=0):
        if(self.logLevel.value < minLevel.value):
            return
        else:
            indentStr = ""
            for _ in range(indent): indentStr += " "

            if self.printToFile:
                with open(self.printFileName, "a") as f:
                    print(indentStr + self.format(messageType), *messages, file = f)

            else:
                print(indentStr + self.format(messageType), *messages)

    def error(self, *messages):
        self.log("ERROR", logLevels.kError, *messages)

    def warn(self, *messages):
        self.log("WARN", logLevels.kWarning, *messages)

    def info(self, *messages):
        self.log("INFO", logLevels.kInfo, *messages)

    def debug(self, *messages):
        self.log("DEBUG", logLevels.kDebug, *messages, indent=1)

    def trace(self, *messages):
        self.log("TRACE", logLevels.kTrace, *messages, indent=2)
    
mainLogger = Logger(name = "main_logger")

def ERROR(*messages):
    mainLogger.error(*messages)

def WARN(*messages):
    mainLogger.warn(*messages)
    
def INFO(*messages):
    mainLogger.info(*messages)
    
def DEBUG(*messages):
    mainLogger.debug(*messages)
    
def TRACE(*messages):
    mainLogger.trace(*messages)
