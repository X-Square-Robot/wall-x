"""
分层级的推理日志系统

层级结构:
- ENV: 环境层 (RealRobotEnv)
- ROBOT: 机器人层 (Robot子类)
- CONTROLLER: 控制器层 (RobotController, RobotCommunication)
- MODEL: 模型层 (WallxModelWrapper)
- UTILS: 工具层 (各种工具类)

使用示例:
    # 方式1: 自动检测层级
    from wall_x.infer.logger import get_logger
    logger = get_logger(__name__)
    logger.info("This is an info message")

    # 方式2: 手动指定层级
    logger = get_logger(__name__, "ROBOT")
    logger.debug("Robot state updated")

    # 方式3: 使用快捷方法
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
    分层级的推理日志系统
    """

    _loggers = {}
    _initialized = False

    # 层级定义
    LEVEL_ENV = "ENV"
    LEVEL_ROBOT = "ROBOT"
    LEVEL_CONTROLLER = "CONTROLLER"
    LEVEL_MODEL = "MODEL"
    LEVEL_UTILS = "UTILS"

    # 层级颜色映射 (用于终端输出)
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
        初始化日志系统

        Args:
            log_level: 日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_dir: 日志文件目录
            console_output: 是否输出到控制台
            file_output: 是否输出到文件
            colorful: 是否使用彩色输出 (需要安装colorlog)
        """
        if cls._initialized:
            return

        cls.log_level = getattr(logging, log_level.upper())
        cls.console_output = console_output
        cls.file_output = file_output
        cls.colorful = colorful and HAS_COLORLOG

        # 创建日志目录
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
        获取指定层级的logger

        Args:
            name: logger名称 (通常是模块名或类名)
            level: 层级标识 (ENV, ROBOT, CONTROLLER, MODEL, UTILS)

        Returns:
            配置好的logger实例
        """
        if not cls._initialized:
            cls.setup()

        # 自动检测层级
        if level is None:
            level = cls._detect_level(name)

        logger_key = f"{level}.{name}"

        if logger_key in cls._loggers:
            return cls._loggers[logger_key]

        # 创建新logger
        logger = logging.getLogger(logger_key)
        logger.setLevel(cls.log_level)
        logger.propagate = False

        # 清除已有handlers
        logger.handlers.clear()

        # 控制台输出
        if cls.console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(cls.log_level)

            if cls.colorful:
                # 彩色格式化
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
                # 普通格式化
                console_format = (
                    f"[%(asctime)s] [{level:^10}] [%(name)s] "
                    f"%(levelname)-8s %(message)s"
                )
                console_formatter = logging.Formatter(
                    console_format, datefmt="%H:%M:%S"
                )

            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)

        # 文件输出
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
        """根据名称自动检测层级"""
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
        """获取环境层logger"""
        return cls.get_logger(name, cls.LEVEL_ENV)

    @classmethod
    def get_robot_logger(cls, name: str = "Robot") -> logging.Logger:
        """获取机器人层logger"""
        return cls.get_logger(name, cls.LEVEL_ROBOT)

    @classmethod
    def get_controller_logger(cls, name: str = "Controller") -> logging.Logger:
        """获取控制器层logger"""
        return cls.get_logger(name, cls.LEVEL_CONTROLLER)

    @classmethod
    def get_model_logger(cls, name: str = "Model") -> logging.Logger:
        """获取模型层logger"""
        return cls.get_logger(name, cls.LEVEL_MODEL)

    @classmethod
    def get_utils_logger(cls, name: str = "Utils") -> logging.Logger:
        """获取工具层logger"""
        return cls.get_logger(name, cls.LEVEL_UTILS)

    @classmethod
    def set_level(cls, level: str):
        """动态修改所有logger的日志级别"""
        new_level = getattr(logging, level.upper())
        cls.log_level = new_level
        for logger in cls._loggers.values():
            logger.setLevel(new_level)
            for handler in logger.handlers:
                handler.setLevel(new_level)

    @classmethod
    def close_all(cls):
        """关闭所有logger的文件句柄"""
        for logger in cls._loggers.values():
            for handler in logger.handlers[:]:
                handler.close()
                logger.removeHandler(handler)
        cls._loggers.clear()
        cls._initialized = False


# 便捷函数
def get_logger(name: str, level: str = None) -> logging.Logger:
    """
    获取logger的便捷函数

    Args:
        name: logger名称 (通常使用 __name__)
        level: 层级标识 (可选,会自动检测)

    Returns:
        配置好的logger实例

    使用示例:
        from wall_x.infer.logger import get_logger
        logger = get_logger(__name__)  # 自动检测层级
        logger = get_logger(__name__, "ROBOT")  # 手动指定层级
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
    设置日志系统的便捷函数

    Args:
        log_level: 日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: 日志文件目录
        console_output: 是否输出到控制台
        file_output: 是否输出到文件
        colorful: 是否使用彩色输出

    使用示例:
        from wall_x.infer.logger import setup_logger
        setup_logger(log_level="DEBUG", log_dir="./logs")
    """
    InferLogger.setup(log_level, log_dir, console_output, file_output, colorful)
