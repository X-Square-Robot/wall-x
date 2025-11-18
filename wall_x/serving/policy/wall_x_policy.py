import logging
from typing import Dict, Any, List
import torch
import numpy as np
from transformers import AutoProcessor

from wall_x.serving.websocket_policy_server import BasePolicy
from wall_x.model.qwen2_5_based.modeling_qwen2_5_vl_act import Qwen2_5_VLMoEForAction
from wall_x.serving.policy.utils import prepare_batch

logger = logging.getLogger(__name__)


class WallXPolicy(BasePolicy):
    """Policy wrapper for Wall-X model that implements the BasePolicy interface."""

    def __init__(
        self,
        model_path: str,
        train_config: dict,
        action_tokenizer_path: str,
        action_dim: int,
        agent_pos_dim: int,
        pred_horizon: int,
        camera_key: List[str],
        device: str = "cuda",
        dtype: str = "bfloat16",
        predict_mode: str = "fast",
        default_prompt: str | None = None,
        min_pixels: int = 4 * 28 * 28,
        max_pixels: int = 16384 * 28 * 28,
        image_factor: int = 28,
        max_length: int = 2048,
    ):
        """Initialize the Wall-X policy.

        Args:
            model_path: Path to the pretrained model checkpoint
            action_tokenizer_path: Path to the action tokenizer
            action_dim: Dimension of action space
            pred_horizon: Prediction horizon for actions
            device: Device to run model on ('cuda' or 'cpu')
            dtype: Data type for model ('bfloat16', 'float16', or 'float32')
            predict_mode: Prediction mode ('fast' or 'slow')
            default_prompt: Default text prompt for the model
            min_pixels: Minimum pixels for image resizing
            max_pixels: Maximum pixels for image resizing
            image_factor: Factor for smart resize
            max_length: Maximum sequence length for text
        """
        logger.info(f"Loading Wall-X model from {model_path}")

        self.model = Qwen2_5_VLMoEForAction.from_pretrained(
            model_path,
            train_config=train_config,
            action_tokenizer_path=action_tokenizer_path,
        )
        self.model.eval()
        self.model = self.model.to(device)

        self.model = self.model.bfloat16()

        # hard code the action dim to 20 for align to wall-x configuration
        self.fixed_action_dim = 20

        self.action_dim = action_dim
        self.agent_pos_dim = agent_pos_dim
        self.pred_horizon = pred_horizon
        self.device = device
        self.predict_mode = predict_mode
        self.default_prompt = default_prompt
        self.camera_key = camera_key

        # Image preprocessing config
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.image_factor = image_factor
        self.max_length = max_length

        # Load processor
        logger.info("Loading processor and tokenizer...")
        self.processor = AutoProcessor.from_pretrained(model_path, use_fast=True)
        self.processor.tokenizer.padding_side = "left"

        # Action buffer for multi-step predictions
        self.action_buffer = []
        self.buffer_index = 0

        logger.info(
            f"Model loaded successfully. Device: {device}, Action dim: {action_dim}, Horizon: {pred_horizon}"
        )

    @property
    def metadata(self) -> Dict[str, Any]:
        """Return metadata about the policy."""
        return {
            "action_dim": self.action_dim,
            "pred_horizon": self.pred_horizon,
            "device": self.device,
            "predict_mode": self.predict_mode,
        }

    def reset(self) -> None:
        """Reset the policy state."""
        self.action_buffer = []
        self.buffer_index = 0
        logger.debug("Policy reset")

    def infer(self, obs: Dict) -> Dict:
        """Infer action from observation.

        Args:
            obs: Dictionary containing:
                - 'image': Image observation (numpy array or PIL Image)
                - 'prompt': Optional text prompt
                - 'state': Optional robot state
                - Other modality-specific observations

        Returns:
            Dictionary containing:
                - 'action': Predicted action (numpy array)
                - Additional metadata
        """
        try:
            # Need to predict new actions
            input_batch = prepare_batch(
                obs,
                self.processor,
                self.camera_key,
                self.agent_pos_dim,
                self.action_dim,
                self.pred_horizon,
                self.fixed_action_dim,
                self.max_length,
                self.image_factor,
                self.min_pixels,
                self.max_pixels,
                self.predict_mode,
                self.device,
            )

            with torch.no_grad():
                outputs = self.model(
                    **input_batch,
                    action_dim=(
                        self.action_dim
                        if self.predict_mode == "fast"
                        else self.fixed_action_dim
                    ),
                    pred_horizon=self.pred_horizon,
                    mode="predict",
                    predict_mode=self.predict_mode,
                )

            if outputs["predict_action"] is None:
                predicted_actions = np.zeros(
                    [1, self.pred_horizon, self.action_dim]
                ).astype(np.float32)

            predicted_actions = (
                outputs["predict_action"][:, :, : self.action_dim]
                .detach()
                .cpu()
                .to(torch.float32)
                .numpy()
            )

            print(predicted_actions.shape)
            return {"action": predicted_actions}

        except Exception as e:
            logger.error(f"Error during inference: {e}")
            raise
