"""
Hierarchical Inference Logging System

Level structure:
- ENV: Environment layer (RealRobotEnv)
- ROBOT: Robot layer (Robot subclasses)
- CONTROLLER: Controller layer (RobotController, RobotCommunication)
- MODEL: Model layer (WallxModelWrapper)
- UTILS: Utility layer (various utility classes)

Usage examples:
    # Method 1: Auto-detect level
    from wall_x.infer.logger import get_logger
    logger = get_logger(__name__)
    logger.info("This is an info message")

    # Method 2: Manually specify level
    logger = get_logger(__name__, "ROBOT")
    logger.debug("Robot state updated")

    # Method 3: Use shortcut methods
    from wall_x.infer.logger import InferLogger
    logger = InferLogger.get_robot_logger("DesktopRobot")
    logger.warning("Action out of bounds")
"""

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
    print("[WARNING] colorlog not installed. Install with: pip install colorlog")


class InferLogger:
    """
    Hierarchical inference logging system
    """

    _loggers = {}
    _initialized = False

    # Level definitions
    LEVEL_ENV = "ENV"
    LEVEL_ROBOT = "ROBOT"
    LEVEL_CONTROLLER = "CONTROLLER"
    LEVEL_MODEL = "MODEL"
    LEVEL_UTILS = "UTILS"

    # Level color mapping (for terminal output)
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
            log_level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_dir: Log file directory
            console_output: Whether to output to console
            file_output: Whether to output to file
            colorful: Whether to use colorful output (requires colorlog)
        """
        if cls._initialized:
            return

        cls.log_level = getattr(logging, log_level.upper())
        cls.console_output = console_output
        cls.file_output = file_output
        cls.colorful = colorful and HAS_COLORLOG

        # Create log directory
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
        Get logger for specified level

        Args:
            name: Logger name (usually module name or class name)
            level: Level identifier (ENV, ROBOT, CONTROLLER, MODEL, UTILS)

        Returns:
            Configured logger instance
        """
        if not cls._initialized:
            cls.setup()

        # Auto-detect level
        if level is None:
            level = cls._detect_level(name)

        logger_key = f"{level}.{name}"

        if logger_key in cls._loggers:
            return cls._loggers[logger_key]

        # Create new logger
        logger = logging.getLogger(logger_key)
        logger.setLevel(cls.log_level)
        logger.propagate = False

        # Clear existing handlers
        logger.handlers.clear()

        # Console output
        if cls.console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(cls.log_level)

            if cls.colorful:
                # Colorful formatting
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
                # Plain formatting
                console_format = (
                    f"[%(asctime)s] [{level:^10}] [%(name)s] "
                    f"%(levelname)-8s %(message)s"
                )
                console_formatter = logging.Formatter(
                    console_format, datefmt="%H:%M:%S"
                )

            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)

        # File output
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
        """Auto-detect level based on name"""
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
        """Get environment layer logger"""
        return cls.get_logger(name, cls.LEVEL_ENV)

    @classmethod
    def get_robot_logger(cls, name: str = "Robot") -> logging.Logger:
        """Get robot layer logger"""
        return cls.get_logger(name, cls.LEVEL_ROBOT)

    @classmethod
    def get_controller_logger(cls, name: str = "Controller") -> logging.Logger:
        """Get controller layer logger"""
        return cls.get_logger(name, cls.LEVEL_CONTROLLER)

    @classmethod
    def get_model_logger(cls, name: str = "Model") -> logging.Logger:
        """Get model layer logger"""
        return cls.get_logger(name, cls.LEVEL_MODEL)

    @classmethod
    def get_utils_logger(cls, name: str = "Utils") -> logging.Logger:
        """Get utility layer logger"""
        return cls.get_logger(name, cls.LEVEL_UTILS)

    @classmethod
    def set_level(cls, level: str):
        """Dynamically modify log level for all loggers"""
        new_level = getattr(logging, level.upper())
        cls.log_level = new_level
        for logger in cls._loggers.values():
            logger.setLevel(new_level)
            for handler in logger.handlers:
                handler.setLevel(new_level)

    @classmethod
    def close_all(cls):
        """Close file handles for all loggers"""
        for logger in cls._loggers.values():
            for handler in logger.handlers[:]:
                handler.close()
                logger.removeHandler(handler)
        cls._loggers.clear()
        cls._initialized = False


# Convenience functions
def get_logger(name: str, level: str = None) -> logging.Logger:
    """
    Convenience function to get logger

    Args:
        name: Logger name (usually use __name__)
        level: Level identifier (optional, will auto-detect)

    Returns:
        Configured logger instance

    Usage examples:
        from wall_x.infer.logger import get_logger
        logger = get_logger(__name__)  # Auto-detect level
        logger = get_logger(__name__, "ROBOT")  # Manually specify level
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
    Convenience function to setup logging system

    Args:
        log_level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Log file directory
        console_output: Whether to output to console
        file_output: Whether to output to file
        colorful: Whether to use colorful output

    Usage examples:
        from wall_x.infer.logger import setup_logger
        setup_logger(log_level="DEBUG", log_dir="./logs")
    """
    InferLogger.setup(log_level, log_dir, console_output, file_output, colorful)
