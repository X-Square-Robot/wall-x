"""Layered logging helpers for inference components."""

import logging
import sys
from typing import Optional
from pathlib import Path
from datetime import datetime

try:
    import colorlog

    HAS_COLORLOG = True
except ImportError:
    HAS_COLORLOG = False
    logging.getLogger(__name__).warning(
        "colorlog not installed. Install with: pip install colorlog"
    )


class InferLogger:
    """
    Hierarchical inference logging system
    """

    _loggers = {}
    _initialized = False

    # layer definitions
    LEVEL_ENV = "ENV"
    LEVEL_ROBOT = "ROBOT"
    LEVEL_CONTROLLER = "CONTROLLER"
    LEVEL_MODEL = "MODEL"
    LEVEL_UTILS = "UTILS"

    # layer colors (terminal output)
    LEVEL_COLORS = {
        LEVEL_ENV: "cyan",
        LEVEL_ROBOT: "green",
        LEVEL_CONTROLLER: "yellow",
        LEVEL_MODEL: "purple",  # colorlog uses 'purple' not 'magenta'
        LEVEL_UTILS: "blue",
    }

    @classmethod
    def setup(
        cls,
        log_level: str = "INFO",
        log_dir: Optional[str] = None,
        console_output: bool = True,
        file_output: bool = True,
        colorful: bool = True,
    ):
        """
        Initialize the logging system

        Args:
            log_level: log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_dir: log file directory
            console_output: emit logs to console
            file_output: emit logs to file
            colorful: colored output (requires colorlog)
        """
        if cls._initialized:
            return

        cls.log_level = getattr(logging, log_level.upper())
        cls.console_output = console_output
        cls.file_output = file_output
        cls.colorful = colorful and HAS_COLORLOG

        # create log directory
        if file_output and log_dir:
            cls.log_dir = Path(log_dir)
            cls.log_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            cls.log_file = cls.log_dir / f"infer_{timestamp}.log"
        else:
            cls.log_file = None

        cls._initialized = True

    @classmethod
    def get_logger(cls, name: str, level: str = None) -> logging.Logger:
        """
        Get a logger for the given layer

        Args:
            name: logger name (usually module or class)
            level: layer tag (ENV, ROBOT, CONTROLLER, MODEL, UTILS)

        Returns:
            Configured logger instance
        """
        if not cls._initialized:
            cls.setup()

        # auto-detect layer
        if level is None:
            level = cls._detect_level(name)

        logger_key = f"{level}.{name}"

        if logger_key in cls._loggers:
            return cls._loggers[logger_key]

        # create new logger
        logger = logging.getLogger(logger_key)
        logger.setLevel(cls.log_level)
        logger.propagate = False

        # clear existing handlers
        logger.handlers.clear()

        # console handler
        if cls.console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(cls.log_level)

            if cls.colorful:
                # colored formatter
                color = cls.LEVEL_COLORS.get(level, "white")
                console_format = (
                    f"%(log_color)s[%(asctime)s]%(reset)s "
                    f"%(bold_{color})s[{level:^10}]%(reset)s "
                    f"%(bold_white)s[%(name)s]%(reset)s "
                    f"%(log_color)s%(levelname)-8s%(reset)s "
                    f"%(message)s"
                )

                console_formatter = colorlog.ColoredFormatter(
                    console_format,
                    datefmt="%H:%M:%S",
                    log_colors={
                        "DEBUG": "cyan",
                        "INFO": "green",
                        "WARNING": "yellow",
                        "ERROR": "red",
                        "CRITICAL": "bold_red",
                    },
                )
            else:
                # plain formatter
                console_format = (
                    f"[%(asctime)s] [{level:^10}] [%(name)s] "
                    f"%(levelname)-8s %(message)s"
                )
                console_formatter = logging.Formatter(
                    console_format, datefmt="%H:%M:%S"
                )

            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)

        # file handler
        if cls.file_output and cls.log_file:
            file_handler = logging.FileHandler(cls.log_file, encoding="utf-8")
            file_handler.setLevel(cls.log_level)

            file_format = (
                f"[%(asctime)s] [{level:^10}] [%(name)s] "
                f"%(levelname)-8s %(message)s"
            )
            file_formatter = logging.Formatter(file_format, datefmt="%Y-%m-%d %H:%M:%S")
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)

        cls._loggers[logger_key] = logger
        return logger

    @classmethod
    def _detect_level(cls, name: str) -> str:
        """Auto-detect layer from logger name."""
        name_lower = name.lower()

        if "env" in name_lower:
            return cls.LEVEL_ENV
        elif "robot" in name_lower and "controller" not in name_lower:
            return cls.LEVEL_ROBOT
        elif (
            "controller" in name_lower
            or "communication" in name_lower
            or "socket" in name_lower
        ):
            return cls.LEVEL_CONTROLLER
        elif "model" in name_lower or "wrapper" in name_lower:
            return cls.LEVEL_MODEL
        else:
            return cls.LEVEL_UTILS

    @classmethod
    def get_env_logger(cls, name: str = "Environment") -> logging.Logger:
        """ENV layer logger."""
        return cls.get_logger(name, cls.LEVEL_ENV)

    @classmethod
    def get_robot_logger(cls, name: str = "Robot") -> logging.Logger:
        """ROBOT layer logger."""
        return cls.get_logger(name, cls.LEVEL_ROBOT)

    @classmethod
    def get_controller_logger(cls, name: str = "Controller") -> logging.Logger:
        """CONTROLLER layer logger."""
        return cls.get_logger(name, cls.LEVEL_CONTROLLER)

    @classmethod
    def get_model_logger(cls, name: str = "Model") -> logging.Logger:
        """MODEL layer logger."""
        return cls.get_logger(name, cls.LEVEL_MODEL)

    @classmethod
    def get_utils_logger(cls, name: str = "Utils") -> logging.Logger:
        """UTILS layer logger."""
        return cls.get_logger(name, cls.LEVEL_UTILS)

    @classmethod
    def set_level(cls, level: str):
        """Change log level for all loggers."""
        new_level = getattr(logging, level.upper())
        cls.log_level = new_level
        for logger in cls._loggers.values():
            logger.setLevel(new_level)
            for handler in logger.handlers:
                handler.setLevel(new_level)

    @classmethod
    def close_all(cls):
        """Close file handles for all loggers."""
        for logger in cls._loggers.values():
            for handler in logger.handlers[:]:
                handler.close()
                logger.removeHandler(handler)
        cls._loggers.clear()
        cls._initialized = False


# convenience helpers
def get_logger(name: str, level: str = None) -> logging.Logger:
    """
    Convenience wrapper to get a logger

    Args:
        name: logger name (usually __name__)
        level: layer tag (optional, auto-detected)

    Returns:
        Configured logger instance

    Example:
        from wall_x._vendor.harrix.serving._wallx_infer.logger import get_logger
        logger = get_logger(__name__)  # auto-detect layer
        logger = get_logger(__name__, "ROBOT")  # manual layer
    """
    return InferLogger.get_logger(name, level)


def setup_logger(
    log_level: str = "INFO",
    log_dir: Optional[str] = None,
    console_output: bool = True,
    file_output: bool = True,
    colorful: bool = True,
):
    """
    Convenience wrapper to configure logging

    Args:
        log_level: log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: log file directory
        console_output: emit logs to console
        file_output: emit logs to file
        colorful: colored output

    Example:
        from wall_x._vendor.harrix.serving._wallx_infer.logger import setup_logger
        setup_logger(log_level="DEBUG", log_dir="./logs")
    """
    InferLogger.setup(log_level, log_dir, console_output, file_output, colorful)
