from enum import Enum
import datetime

logLevels = Enum("logLevels", "kSilent kError kWarning kInfo kDebug kTrace")

## set global default log level
LOG_LEVEL = logLevels.kTrace

class Logger:
    nLoggers = 0
    def __init__(self, logLevel = LOG_LEVEL, name = ""):
        self.logLevel = logLevel
        if(name == ""):
            self.name = "Logger_" + str(Logger.nLoggers)
        else:
            self.name = name

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



