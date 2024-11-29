import logging

class ModuleLogger:
    __module_logger = None
    
    @classmethod
    def get_logger(cls) -> logging.Logger:
        """get IRL module logger"""
        if cls.__module_logger is None:
            cls.__module_logger = logging.getLogger("IRL")
            
            # initialize module logger
            cls.__module_logger.setLevel(logging.WARNING)
        return cls.__module_logger

get_module_logger = ModuleLogger.get_logger