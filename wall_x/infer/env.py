"""
Base Environment Class for Robot Control and Inference
"""

from typing import Dict, Any, List
from abc import ABC, abstractmethod
import time
from wall_x.infer.infer_config import InferConfig
from wall_x.infer.utils import KeyboardThread
from wall_x.infer.logger import InferLogger


class BaseEnv(ABC):
    def __init__(self, config: InferConfig):
        self.config = config
        self.logger = InferLogger.get_env_logger("Env")

    @abstractmethod
    def get_observation(self) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def apply_action(self, input: dict) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_instruction(self) -> str:
        raise NotImplementedError

    def reset(self) -> Dict[str, Any]:
        raise NotImplementedError

    def stop(self) -> None:
        raise NotImplementedError


class RealRobotEnv(BaseEnv):
    def __init__(
        self, config: InferConfig, instructions: List[str], enable_keyboard: bool = True
    ):
        """
        Args:
            config: Inference configuration
            instruction: Task instruction
        """
        super().__init__(config)
        self.instruction = "test"
        self.model = self._register_model()
        self.robot = self._register_robot()

        # Keyboard control
        self.keyboard_thread = None
        if enable_keyboard:
            self.keyboard_thread = KeyboardThread()

        # Instruction list
        self.instructions = instructions
        self.instruction_index = 0

    # def _register_model(self) -> WallxModelWrapper:
    #     return WallxModelWrapper(self.config)

    def _register_robot(self):
        from wall_x.infer.robot import DesktopRobot, TurtleRobot

        if self.config.robot_type == "desktop":
            return DesktopRobot(self.config)
        elif self.config.robot_type == "turtle":
            return TurtleRobot(self.config)
        else:
            raise ValueError(f"Invalid robot type: {self.config.robot_type}")

    def get_observation(self):
        return self.robot.get_observation()

    def apply_action(self, input: dict):
        self.robot.apply_action(input)

    def get_instruction(self) -> str:
        """Return task instruction"""
        return self.instructions[self.instruction_index]

    def reset(self):
        self.robot.go_home()

    def listen_to_keyboard(self):
        if self.keyboard_thread is not None:
            if self.keyboard_thread.should_stop:
                time.sleep(1)
                return True
            if self.keyboard_thread.should_reset:
                self.reset()
                self.keyboard_thread.should_reset = False
                time.sleep(1)
                return True
            if self.keyboard_thread.new_instruction_index is not None:
                new_index = self.keyboard_thread.new_instruction_index
                # Check if index is valid
                if 0 <= new_index < len(self.instructions):
                    self.instruction_index = new_index
                    self.logger.info(
                        f"[Keyboard] Instruction index switched to {new_index}: {self.instructions[new_index]}"
                    )
                else:
                    self.logger.info(
                        f"[Keyboard] Invalid instruction index {new_index}, valid range: 0-{len(self.instructions)-1}"
                    )
                # Reset flag
                self.keyboard_thread.new_instruction_index = None
                time.sleep(1)
                return True
        return False

    def run_infer_flow_action(self):
        while True:
            if self.listen_to_keyboard():
                continue
            observation = self.get_observation()
            instruction = self.get_instruction()
            model_output = self.model.infer_flow_action(observation, instruction)
            self.apply_action(model_output)

    def run_infer_flow_action_with_subtask(self, subtask_interval: int = 2):
        step = 0
        subtask = ""
        while True:
            if self.listen_to_keyboard():
                continue
            observation = self.get_observation()
            instruction = self.get_instruction()
            if step == 0 or step % subtask_interval == 0:
                subtask = self.model.infer_subtask(observation, instruction)
            model_output = self.model.infer_flow_action(observation, subtask)
            self.apply_action(model_output)

    def run_infer_ar_action(self):
        while True:
            if self.listen_to_keyboard():
                continue
            observation = self.get_observation()
            instruction = self.get_instruction()
            model_output = self.model.infer_ar_action(observation, instruction)
            self.apply_action(model_output)
