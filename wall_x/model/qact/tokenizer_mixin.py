"""Action tokenizer mixin for loading tokenizers and action mappings."""

import torch
import numpy as np
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
from transformers import AutoProcessor

from wall_x.utils.constant import is_action_dataset_name

# Delay imports so missing optional packages do not fail at import time
try:
    from spatial_tokenizer.spatial_tokenizer import SpatialActionTokenizer
except ImportError:
    SpatialActionTokenizer = None

try:
    from x2robot_tokenizer.x2robot_tokenizer_v3_1_delta.api.tokenizer import (
        X2RobotTokenizerV3_1,
    )
except ImportError:
    X2RobotTokenizerV3_1 = None


class ActionTokenizerMixin(ABC):
    """Base class for action tokenizers"""

    def __init__(self):
        self._tokenizer = None
        self.action_normalizer = None
        self._tokenizer_type: str = ""
        self._action_mapper_cache: Optional[Dict] = None  # Cached action_mapper
        self.dllm = False
        self.input_placeholder_flag = False

    @property
    def tokenizer_type(self) -> str:
        """Return the tokenizer type identifier"""
        return self._tokenizer_type

    @property
    def tokenizer(self):
        """Return the underlying tokenizer instance"""
        return self._tokenizer

    @property
    def action_mapper(self) -> Optional[Dict]:
        """Return the cached action_mapper"""
        return self._action_mapper_cache

    @abstractmethod
    def load_tokenizer(self, config: dict, normalizer, device: str = "cpu") -> Any:
        """
        Load the tokenizer instance

        Args:
            config: Configuration dictionary
            normalizer: Normalizer
            device: Device, usually "cpu" for training and "cuda" for inference

        Returns:
            tokenizer instance
        """
        pass

    @abstractmethod
    def get_val_tokenizer(self, config: dict) -> Any:
        """
        Get the tokenizer used for validation/inference

        Args:
            config: Configuration dictionary

        Returns:
            validation tokenizer instance
        """
        pass

    @abstractmethod
    def get_special_tokens(self) -> List[str]:
        """
        Return special tokens to add to the vocabulary

        Returns:
            list of token strings
        """
        pass

    def get_all_special_tokens(self) -> Tuple[List[str], Optional[List[str]]]:
        """
        Return all special tokens and keep <|action_token_0|> before AR action tokens

        Returns:
            (new_tokens, special_tokens) tuple
        """
        tokens, special_tokens = self.get_special_tokens()
        if "<|action_token_0|>" not in tokens:
            tokens.insert(0, "<|action_token_0|>")
        return tokens, special_tokens

    @abstractmethod
    def build_action_mapper(self, processor) -> Optional[Dict]:
        """
        Build action_mapper

        Args:
            processor: HuggingFace processor，used to convert tokens to IDs

        Returns:
            action_mapper dictionary; format depends on tokenizer type
        """
        pass

    @abstractmethod
    def get_action_token_list(self, processor) -> List[int]:
        """
        Get the action token ID list

        Args:
            processor: HuggingFace processor

        Returns:
            action token ID list
        """
        pass

    @abstractmethod
    def decode_action(
        self,
        output_ids: torch.Tensor,
        action_mapper: Dict,
        action_horizon: int,
        action_dim: int,
        device: torch.device,
        proprioception: Optional[torch.Tensor] = None,
        dof_mask: Optional[torch.Tensor] = None,
        robot_type_id: Optional[int] = None,
        state: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[Union[np.ndarray, torch.Tensor]], bool]:
        """
        Unified decoding interface

        Args:
            output_ids: model output token IDs [1, seq_len]
            action_mapper: action_mapper dictionary
            action_horizon: action horizon
            action_dim: action dimension
            device: Device
            proprioception: proprioception, normalized when required by a tokenizer
            dof_mask: DOF mask when required by a tokenizer
            robot_type_id: robot type ID when required by a tokenizer
            state: state when required by fast/spatial tokenizers

        Returns:
            (predict_action, decode_success)
            - predict_action: decoded action [T, action_dim] or None
            - decode_success: whether decoding succeeded
        """
        pass

    @abstractmethod
    def compute_accuracy(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        action_mapper: Dict,
        action_token_id_set: Dict,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute accuracy metrics

        Args:
            logits: model output logits
            labels: Labels
            action_mapper: action_mapper dictionary
            action_token_id_set: set of action token IDs

        Returns:
            accuracy metric dictionary, such as {"action_accuracy": tensor, ...}
        """
        pass

    @abstractmethod
    def get_accuracy_keys(self) -> List[str]:
        """
        Return accuracy metric keys for logging

        Returns:
            list of key strings
        """
        pass

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        """Return the vocabulary size"""
        pass

    @property
    @abstractmethod
    def uses_dof_mask_for_unnorm(self) -> bool:
        """
        Whether unnormalization requires dof_mask

        Returns:
            True: dof_mask is required by fast/spatial tokenizers
            False: dof_mask is not required and full dimensions are returned
        """
        pass

    @property
    @abstractmethod
    def needs_action_crop(self) -> bool:
        """
        Whether actions must be clipped before encoding

        Returns:
            True: clip by chunk_size and dof_mask for fast/spatial tokenizers
            False: no clipping; the encoder handles the full sequence internally
        """
        pass

    @abstractmethod
    def encode_to_tokens(
        self,
        actions: torch.Tensor,
        obs_state: Optional[torch.Tensor] = None,
        dof_mask: Optional[torch.Tensor] = None,
        robot_type_ids: Optional[List[int]] = None,
        is_train: bool = True,
    ) -> List[List[str]]:
        """
        Encode actions into token strings for training data processing

        Args:
            actions: Normalized actions
                - fast/spatial: List[Tensor], each [T, D] (clipped)
                - v3.1 delta: Tensor [B, T, D] (full sequence)
            obs_state: observation state when required by a tokenizer[B, obs_horizon, D]
            dof_mask: DOF mask[B, T, D]
            robot_type_ids: robot type ID list when required by a tokenizer

        Returns:
            List[List[str]]: token string list for each sample
        """
        pass

    def init_inference(self, robot_type: Optional[str] = None) -> None:
        """
        Inference initialization hook; subclasses may override

        Args:
            robot_type: robot type name
        """
        pass

    def prepare_action_for_ar_encoding(
        self,
        ar_actionchunk: torch.Tensor,
        dataset_names: List[str],
        agent_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Prepare action data before AR encoding; subclasses may override

        SpatialVLA needs waypoint selection; other tokenizers return normalized_action directly

        Args:
            ar_actionchunk: [B, T, D] actions before normalization
            dataset_names: [str] dataset name list
            agent_pos: [B, T, D] agent positions required by SpatialVLA

        Returns:
            prepared action data
        """
        if self.action_normalizer is not None:
            ar_actionchunk = self.action_normalizer.normalize_data(
                ar_actionchunk, dataset_names
            )
        return ar_actionchunk

    def get_robot_type_ids(
        self,
        uids: List[Optional[str]],
        dataset_names: List[str],
    ) -> Optional[List[int]]:
        """
        Get robot_type_ids; subclasses may override

        Only the v3.1 delta tokenizer needs this; other tokenizers return None

        Args:
            uids: UID list
            dataset_names: dataset name list

        Returns:
            robot_type_id list or None
        """
        return None

    def modify_inputs_for_dllm(
        self,
        inputs: Dict[str, torch.Tensor],
        processor,
        sample_time: torch.Tensor,
        dataset_names: List[str],
    ):
        return NotImplementedError


class FastTokenizerMixin(ActionTokenizerMixin):
    """Fast tokenizer implementation"""

    def __init__(self):
        super().__init__()
        self._tokenizer_type = "fast"

    def load_tokenizer(self, config: dict, normalizer, device: str = "cpu") -> Any:
        """Load the fast tokenizer"""
        self._tokenizer = AutoProcessor.from_pretrained(
            config["action_tokenizer_path"], trust_remote_code=True
        )
        self.action_normalizer = normalizer
        return self._tokenizer

    def get_val_tokenizer(self, config: dict) -> Any:
        return self._tokenizer

    def get_special_tokens(self) -> List[str]:
        """Return special tokens for the fast tokenizer"""
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer not loaded. Call load_tokenizer first.")
        # tokens = ["<|ar_action|>", "<|ar_pad|>"] #temporary compatibility with existing checkpoints
        tokens = []
        for i in range(self._tokenizer.vocab_size):
            tokens.append(f"<|action_token_{i}|>")
        return tokens, None

    def build_action_mapper(self, processor) -> Dict[int, int]:
        """
        Build the cached action_mapper for the fast tokenizer

        Returns:
            Dict[token_id, action_idx]
        """
        # Return cached value if available
        if self._action_mapper_cache is not None:
            return self._action_mapper_cache

        if self._tokenizer is None:
            raise RuntimeError("Tokenizer not loaded. Call load_tokenizer first.")

        action_mapper = {}
        for i in range(self._tokenizer.vocab_size):
            token = f"<|action_token_{i}|>"
            token_id = processor.tokenizer.convert_tokens_to_ids(token)
            action_mapper[token_id] = i

        self._action_mapper_cache = action_mapper
        return action_mapper

    def get_action_token_list(self, processor) -> List[int]:
        """Get the action token ID list"""
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer not loaded. Call load_tokenizer first.")

        action_token_list = []
        for i in range(self._tokenizer.vocab_size):
            token_id = processor.tokenizer.convert_tokens_to_ids(
                f"<|action_token_{i}|>"
            )
            action_token_list.append(token_id)
        return action_token_list

    def decode_action(
        self,
        output_ids: torch.Tensor,
        action_mapper: Dict,
        action_horizon: int,
        action_dim: int,
        device: torch.device,
        proprioception: Optional[torch.Tensor] = None,
        dof_mask: Optional[torch.Tensor] = None,
        robot_type_id: Optional[int] = None,
        state: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[Union[np.ndarray, torch.Tensor]], bool]:
        """Fast tokenizer decoding"""
        action_id = []
        for token_id_i in output_ids[0]:
            if token_id_i.item() in action_mapper:
                action_id.append(action_mapper[token_id_i.item()])

        if len(action_id) == 0:
            return np.zeros((action_horizon, action_dim)), False

        predict_action = self._tokenizer.decode(
            [action_id], time_horizon=action_horizon, action_dim=action_dim
        )

        # Check whether decoding succeeded
        decode_success = False
        if isinstance(predict_action, np.ndarray):
            decode_success = np.sum(predict_action) != 0
        elif isinstance(predict_action, torch.Tensor):
            decode_success = predict_action.sum().item() != 0

        return predict_action, decode_success

    def compute_accuracy(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        action_mapper: Dict,
        action_token_id_set: Dict,
    ) -> Dict[str, torch.Tensor]:
        """Compute fast tokenizer accuracy"""
        result = {}

        if len(action_token_id_set.get("action_token_list", [])) > 0:
            shift_logits = logits[..., :-1, :].contiguous()
            action_preds = shift_logits.argmax(dim=-1)
            shift_labels = labels[..., 1:].contiguous()
            action_mask = shift_labels > action_token_id_set["action_token_list"][0]
            correct_preds = (action_preds == shift_labels) & action_mask
            action_accuracy = correct_preds.sum().float() / action_mask.sum().float()
            result["action_accuracy"] = action_accuracy

        return result

    def get_accuracy_keys(self) -> List[str]:
        """Return fast tokenizer accuracy keys"""
        return ["action_accuracy"]

    @property
    def vocab_size(self) -> int:
        if self._tokenizer is None:
            return 0
        return self._tokenizer.vocab_size

    @property
    def uses_dof_mask_for_unnorm(self) -> bool:
        """fast tokenizer requires dof_mask"""
        return True

    @property
    def needs_action_crop(self) -> bool:
        return True

    @property
    def inference_ar_steps_for_dllm(self) -> int:
        return self.max_length

    def encode_to_tokens(
        self,
        actions: List,
        obs_state: Optional[torch.Tensor] = None,
        dof_mask: Optional[torch.Tensor] = None,
        robot_type_ids: Optional[List[int]] = None,
        is_train: bool = True,
    ) -> List[List[str]]:
        """
        Fast tokenizer encoding

        Args:
            actions: List[Tensor/ndarray], each [T, D] (clipped)
        """
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer not loaded. Call load_tokenizer first.")

        all_action_tokens = []
        for i in range(len(actions)):
            action = actions[i]
            if isinstance(action, torch.Tensor):
                action = action.cpu().numpy()
            token_id = self._tokenizer(action)
            action_tokens = [f"<|action_token_{idx}|>" for idx in token_id[0]]
            all_action_tokens.append(action_tokens)
        return all_action_tokens

    def modify_inputs_for_dllm(
        self,
        inputs: Dict[str, torch.Tensor],
        processor,
        sample_time: torch.Tensor,
        dataset_names: List[str],
    ):
        # Untested
        input_ids = inputs["input_ids"]
        labels = inputs["labels"]
        prefix_length = inputs["prefix_length"]
        bs, seqlen = input_ids.shape

        ar_token_length = self.max_length
        ar_step_num = ar_token_length

        device = input_ids.device
        dtype = input_ids.dtype

        placeholder_ids = torch.tensor(
            processor.placeholder_seq, device=device, dtype=dtype
        )

        if not torch.is_tensor(sample_time):
            sample_time = torch.tensor(sample_time, device=device, dtype=torch.float32)
        else:
            sample_time = sample_time.to(device=device, dtype=torch.float32)

        sample_time = sample_time.clamp(0.0, 1.0)

        noisy_steps_per_sample = (
            torch.ceil((1.0 - sample_time) * (ar_step_num + 1)).long() - 1
        )
        noisy_steps_per_sample = noisy_steps_per_sample.clamp(min=0, max=ar_step_num)
        start = prefix_length - ar_token_length - 2
        end = prefix_length - 2

        # Prepare the noise sequence for each sample
        noise_seqs = placeholder_ids.repeat(bs, 1)

        ar_len = end - start
        if noise_seqs.size(1) != ar_len:
            # These should usually match; defensively truncate to ar_len
            noise_seqs = noise_seqs[:, :ar_len]
        rand = torch.rand(bs, ar_len, device=device)
        perm = rand.argsort(dim=-1)
        ranks = perm.argsort(dim=-1)
        noisy_mask = ranks < noisy_steps_per_sample.view(-1, 1)

        ar_input = input_ids[:, start:end]
        ar_labels = labels[:, start + 1 : end + 1]
        ar_input[noisy_mask] = noise_seqs[noisy_mask]
        ar_labels[~noisy_mask] = -100

        inputs["input_ids"] = input_ids
        inputs["labels"] = labels
        return inputs

    def update_placeholder_mask(self, processor, prefix_length, input_ids):
        # Untested
        ar_action_mask = torch.zeros_like(input_ids)

        inc = torch.arange(
            1,
            self.max_length + 1,
        )
        inc = inc.unsqueeze(0).expand(ar_action_mask.size(0), -1)  # [bs, ar_len]
        ar_action_mask[
            :,
            prefix_length - self.max_length - 2 : prefix_length - 2,
        ] = inc
        ar_action_mask[:, prefix_length - 2 : prefix_length] = -1  # eos

        return {"ar_action_mask": ar_action_mask}

    def get_placeholder_for_dllm(self):
        placeholder_seq = ["<|ar_action|>"] * self.max_length
        return placeholder_seq


class SpatialVLATokenizerMixin(ActionTokenizerMixin):
    """SpatialVLA tokenizer implementation"""

    def __init__(self):
        super().__init__()
        self._tokenizer_type = "spatialvla"

    def load_tokenizer(self, config: dict, normalizer, device: str = "cpu") -> Any:
        """Load the SpatialVLA tokenizer"""
        if SpatialActionTokenizer is None:
            raise ImportError(
                "SpatialActionTokenizer is not installed. "
                "Please install spatial_tokenizer package."
            )
        self._tokenizer = SpatialActionTokenizer(
            normalizer=normalizer,
            augment_ratio=config.get("augment_ratio", 0.0),
            max_waypoints=config.get("max_waypoints", 5),
            with_gripper=config.get("with_gripper", True),
            single_arm=config.get("single_arm", False),
        )
        self._val_tokenizer = None
        self.config = config
        self.dllm = config.get("dllm", False)
        self.input_placeholder_flag = config.get("input_placeholder_flag", False)
        self.action_normalizer = normalizer
        self.with_gripper = config.get("with_gripper", True)
        return self._tokenizer

    def get_placeholder_for_dllm(self):
        if self.with_gripper:
            placeholder_seq = [
                "<|left_xyz|>",
                "<|left_rpy|>",
                "<|left_gripper|>",
                "<|right_xyz|>",
                "<|right_rpy|>",
                "<|right_gripper|>",
            ]
        else:
            placeholder_seq = [
                "<|left_xyz|>",
                "<|left_rpy|>",
                "<|right_xyz|>",
                "<|right_rpy|>",
            ]
        if self._tokenizer.single_arm:
            placeholder_seq = placeholder_seq[len(placeholder_seq) // 2 :]
        placeholder_seq = placeholder_seq * self._tokenizer.max_waypoints
        return placeholder_seq

    def get_val_tokenizer(self, config: dict) -> Any:
        if self._val_tokenizer:
            return self._val_tokenizer

        self._val_tokenizer = SpatialActionTokenizer(
            normalizer=self.action_normalizer,
            augment_ratio=0,
            max_waypoints=self.config.get("max_waypoints", 5),
            with_gripper=self.config.get("with_gripper", True),
            single_arm=self.config.get("single_arm", False),
        )
        return self._val_tokenizer

    def get_special_tokens(self) -> List[str]:
        """Return special tokens for the SpatialVLA tokenizer"""
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer not loaded. Call load_tokenizer first.")
        tokens = [
            "<|step|>",
            "<|left|>",
            "<|right|>",
            "<|move|>",
        ]  # only for compatibility with existing checkpoints
        if self.input_placeholder_flag:
            special_tokens = [
                "<|left_xyz|>",
                "<|left_rpy|>",
                "<|left_gripper|>",
                "<|right_xyz|>",
                "<|right_rpy|>",
                "<|right_gripper|>",
            ]
            tokens += special_tokens
            if not self._tokenizer.with_gripper:
                indices = [0, 1, 3, 4]
                special_tokens = [special_tokens[i] for i in indices]
            if self._tokenizer.single_arm:
                special_tokens = special_tokens[len(special_tokens) // 2 :]
        for i in range(self._tokenizer.vocab_size):
            tokens.append(f"<|action_token_{i}|>")

        return tokens, special_tokens

    def build_action_mapper(self, processor) -> Dict[int, int]:
        """
        Build the cached action_mapper for the SpatialVLA tokenizer

        Returns:
            Dict[token_id, action_idx]
        """
        # Return cached value if available
        if self._action_mapper_cache is not None:
            return self._action_mapper_cache

        if self._tokenizer is None:
            raise RuntimeError("Tokenizer not loaded. Call load_tokenizer first.")

        action_mapper = {}
        for i in range(self._tokenizer.vocab_size):
            token = f"<|action_token_{i}|>"
            token_id = processor.tokenizer.convert_tokens_to_ids(token)
            action_mapper[token_id] = i

        self._action_mapper_cache = action_mapper
        return action_mapper

    def get_action_token_list(self, processor) -> List[int]:
        """Get the action token ID list"""
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer not loaded. Call load_tokenizer first.")

        action_token_list = []
        for i in range(self._tokenizer.vocab_size):
            token_id = processor.tokenizer.convert_tokens_to_ids(
                f"<|action_token_{i}|>"
            )
            action_token_list.append(token_id)
        return action_token_list

    def decode_action(
        self,
        output_ids: torch.Tensor,
        action_mapper: Dict,
        action_horizon: int,
        action_dim: int,
        device: torch.device,
        proprioception: Optional[torch.Tensor] = None,
        dof_mask: Optional[torch.Tensor] = None,
        robot_type_id: Optional[int] = None,
        state: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[Union[np.ndarray, torch.Tensor]], bool]:
        """SpatialVLA tokenizer decoding"""
        action_id = []
        for token_id_i in output_ids[0]:
            if token_id_i.item() in action_mapper:
                action_id.append(action_mapper[token_id_i.item()])

        if len(action_id) == 0:
            return np.zeros((action_horizon, action_dim)), False

        if state is not None:
            predict_action = self._tokenizer.decode(
                [action_id],
                state=state[0, 0, :action_dim],
                time_horizon=action_horizon,
                action_dim=action_dim,
            )
        else:
            predict_action = self._tokenizer.decode(
                [action_id], time_horizon=action_horizon, action_dim=action_dim
            )

        # Check whether decoding succeeded
        decode_success = False
        if isinstance(predict_action, np.ndarray):
            decode_success = np.sum(predict_action) != 0
        elif isinstance(predict_action, torch.Tensor):
            decode_success = predict_action.sum().item() != 0

        return predict_action, decode_success

    def compute_accuracy(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        action_mapper: Dict,
        action_token_id_set: Dict,
    ) -> Dict[str, torch.Tensor]:
        """Compute SpatialVLA tokenizer accuracy, same as fast"""
        result = {}

        if len(action_token_id_set.get("action_token_list", [])) > 0:
            shift_logits = logits[..., :-1, :].contiguous()
            action_preds = shift_logits.argmax(dim=-1)
            shift_labels = labels[..., 1:].contiguous()
            action_mask = shift_labels > action_token_id_set["action_token_list"][0]
            correct_preds = (action_preds == shift_labels) & action_mask
            action_accuracy = correct_preds.sum().float() / action_mask.sum().float()
            result["action_accuracy"] = action_accuracy

        return result

    def get_accuracy_keys(self) -> List[str]:
        """Return SpatialVLA tokenizer accuracy keys"""
        return ["action_accuracy"]

    @property
    def vocab_size(self) -> int:
        if self._tokenizer is None:
            return 0
        return self._tokenizer.vocab_size

    @property
    def uses_dof_mask_for_unnorm(self) -> bool:
        """SpatialVLA tokenizer requires dof_mask"""
        return True

    @property
    def needs_action_crop(self) -> bool:
        return False

    @property
    def inference_ar_steps_for_dllm(self) -> int:
        return self._tokenizer.max_waypoints

    def prepare_action_for_ar_encoding(
        self,
        ar_actionchunk: torch.Tensor,
        dataset_names: List[str],
        agent_pos: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        step = ar_actionchunk.shape[1] // self._tokenizer.max_waypoints
        indices = np.arange(0, ar_actionchunk.shape[1], step)[
            : self._tokenizer.max_waypoints
        ]
        ar_action = ar_actionchunk[:, indices, :]
        ar_action = self.action_normalizer.normalize_data(ar_action, dataset_names)
        return ar_action

    def encode_to_tokens(
        self,
        actions: List,
        obs_state: Optional[torch.Tensor] = None,
        dof_mask: Optional[torch.Tensor] = None,
        robot_type_ids: Optional[List[int]] = None,
        is_train: bool = True,
    ) -> List[List[str]]:
        """
        SpatialVLA tokenizer encoding

        Args:
            actions: List[Tensor/ndarray], each [T, D] (clipped)
        """
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer not loaded. Call load_tokenizer first.")

        if isinstance(actions, torch.Tensor):
            # Convert torch tensors to numpy arrays
            actions = actions.cpu().numpy()
        tokenizer = self._tokenizer if is_train else self.get_val_tokenizer({})
        token_ids_group = tokenizer.batch_encode(actions)
        all_action_tokens = []
        for i in range(len(token_ids_group)):
            token_ids = np.array(token_ids_group[i]).reshape(-1)
            action_token = [f"<|action_token_{i}|>" for i in token_ids]
            all_action_tokens.append(action_token)
        return all_action_tokens

    def modify_inputs_for_dllm(
        self,
        inputs: Dict[str, torch.Tensor],
        processor,
        sample_time: torch.Tensor,
        dataset_names: List[str],
    ):
        """
        Prepare AR DLLM inputs by encoding sample_time into input_ids and labels

        Noise injection strategy:
        - Split [0, 1] into N equal parts (N = ar_step_num)
        - For each sample, compute the number of noisy steps k in [1, N]
        k = clamp(ceil((1 - t) * N), 1, N)
        -> smaller t means more noise; values closer to 1 mean less noise, with at least one noised step
        - seq is the noise sequence formed by concatenating ar_step_num placeholder_seq blocks
        Each step maps to len(placeholder_seq) consecutive tokens
        - Noise injection overwrites matching tokens in input_ids / labels step by step from seq
        """

        input_ids = inputs["input_ids"]
        labels = inputs["labels"]
        prefix_length = inputs["prefix_length"]  # int or tensor(1,)
        bs, seqlen = input_ids.shape

        # Total number of AR steps
        ar_step_num = self._tokenizer.max_waypoints  # N
        step_token_len = len(processor.placeholder_seq)  # Number of tokens per step
        ar_token_length = ar_step_num * step_token_len  # Total AR token length

        device = input_ids.device
        dtype = input_ids.dtype

        placeholder_ids = torch.tensor(
            processor.placeholder_seq, device=device, dtype=dtype
        )  # (step_token_len,)
        noise_seq = placeholder_ids.repeat(ar_step_num)  # (ar_token_length,)

        if not torch.is_tensor(sample_time):
            sample_time = torch.tensor(sample_time, device=device, dtype=torch.float32)
        else:
            sample_time = sample_time.to(device=device, dtype=torch.float32)

        sample_time = sample_time.clamp(0.0, 1.0)

        noisy_steps_per_sample = (
            torch.ceil((1.0 - sample_time) * (ar_step_num + 1)).long() - 1
        )
        noisy_steps_per_sample = noisy_steps_per_sample.clamp(min=0, max=ar_step_num)
        start = prefix_length - ar_token_length - 2
        end = prefix_length - 2

        action_idx = 0  # action sample counter
        for i in range(bs):
            if not is_action_dataset_name(dataset_names[i]):
                continue
            k = noisy_steps_per_sample[action_idx].item()
            action_idx += 1
            perm = torch.randperm(ar_step_num, device=device)
            chosen_steps = perm[:k]  # (k,)

            step_mask = torch.zeros(ar_step_num, dtype=torch.bool, device=device)
            step_mask[chosen_steps] = True

            token_mask = step_mask.repeat_interleave(
                step_token_len
            )  # (ar_token_length,)
            ignore_mask = ~token_mask
            # Take a view of the current sample's AR span for mask-based replacement
            cur_input_view = input_ids[i, start:end]
            cur_label_view = labels[
                i, start + 1 : end + 1
            ]  # shift placeholders one position to the right

            # Overwrite selected tokens with noise
            cur_input_view[token_mask] = noise_seq[token_mask]
            cur_label_view[ignore_mask] = -100

        inputs["input_ids"] = input_ids
        inputs["labels"] = labels
        return inputs

    def update_positional_masks_for_dllm(
        self, positional_masks, inputs, processor, visible_predict_ar_ratio=1
    ):
        if self.dllm and self.input_placeholder_flag:
            mask = self.update_placeholder_mask(
                processor,
                inputs["prefix_length"],
                inputs["input_ids"],
            )
            positional_masks.update(mask)
            if (
                np.random.rand() < visible_predict_ar_ratio
            ):  # FIXME ar_visible is temporarily decided per batch; mixed settings are untested and need optimization.
                positional_masks["ar_visible"] = True
                if "ar_predict_token_positions" in positional_masks:
                    del positional_masks["ar_predict_token_positions"]
                # positional_masks["ar_predict_token_positions"] = None
            else:
                positional_masks["ar_visible"] = False
        return positional_masks

    def update_placeholder_mask(self, processor, prefix_length, input_ids):
        ar_action_mask = torch.zeros_like(input_ids)
        current_index = 1
        step_len = len(processor.placeholder_seq)

        for b, seq in enumerate(input_ids):
            for i in range(self._tokenizer.max_waypoints):
                start_idx = (
                    prefix_length - (self._tokenizer.max_waypoints - i) * step_len - 2
                )
                end_idx = start_idx + step_len
                ar_action_mask[b, start_idx:end_idx] = current_index
                current_index += 1
            ar_action_mask[b, end_idx:prefix_length] = -1  # eos

        return {"ar_action_mask": ar_action_mask}


class X2RobotV31DeltaTokenizerMixin(ActionTokenizerMixin):
    """X2Robot V3.1 Delta tokenizer implementation"""

    def __init__(self):
        super().__init__()
        self._tokenizer_type = "x2robot_tokenizer_v3_1_delta"
        self._robot_type_id: Optional[int] = None

    def load_tokenizer(self, config: dict, normalizer, device: str = "cpu") -> Any:
        """
        Load the x2robot_tokenizer_v3_1_delta tokenizer

        Note: use "cpu" during training to avoid device mismatches in distributed runs
        """
        if X2RobotTokenizerV3_1 is None:
            raise ImportError(
                "X2RobotTokenizerV3_1 is not installed. "
                "Please install x2robot_tokenizer package."
            )
        self._tokenizer = X2RobotTokenizerV3_1(
            checkpoint_path=config["action_tokenizer_checkpoint_path"],
            config_dir=config.get("action_tokenizer_config_dir"),
            device=device,
        )
        self.action_normalizer = normalizer

        # DLLM configuration
        self.config = config
        self.dllm = config.get("dllm", False)
        self.input_placeholder_flag = config.get("input_placeholder_flag", False)

        return self._tokenizer

    def get_val_tokenizer(self, config: dict) -> Any:
        """v3.1 delta uses the same tokenizer instance"""
        return self._tokenizer

    def get_special_tokens(self) -> List[str]:
        """Return special tokens for x2robot_tokenizer_v3_1_delta (RVQ tokens)"""
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer not loaded. Call load_tokenizer first.")

        tokens = []
        codebook_size = self._tokenizer.codebook_size
        num_quantizers = self._tokenizer.num_quantizers

        tokens.append("<rvq_group>")
        for q in range(num_quantizers):
            for idx in range(codebook_size):
                tokens.append(f"<rvq_r{q}_{idx}>")

        # DLLM placeholder tokens
        special_tokens = None
        if getattr(self, "input_placeholder_flag", False):
            special_tokens = ["<rvq_placeholder_group>"]
            for q in range(num_quantizers):
                special_tokens.append(f"<rvq_placeholder_{q}>")
            tokens += special_tokens

        return tokens, special_tokens

    def build_action_mapper(self, processor) -> Dict:
        """
        Build the cached action_mapper for x2robot_tokenizer_v3_1_delta

        Returns:
            Dict contains:
                - type: "x2robot_tokenizer_v3_1_delta"
                - rvq_group_token_id: int
                - token_id_to_rvq: Dict[int, Tuple[int, int]]
                - num_quantizers: int
                - codebook_size: int
                - compression_ratio: int
                - action_dim: int
                - token_range: Dict[str, int]
        """
        # Return cached value if available
        if self._action_mapper_cache is not None:
            return self._action_mapper_cache

        if self._tokenizer is None:
            raise RuntimeError("Tokenizer not loaded. Call load_tokenizer first.")

        codebook_size = self._tokenizer.codebook_size
        num_quantizers = self._tokenizer.num_quantizers
        rvq_group_token_id = processor.tokenizer.convert_tokens_to_ids("<rvq_group>")

        action_mapper = {
            "type": "x2robot_tokenizer_v3_1_delta",
            "rvq_group_token_id": rvq_group_token_id,
            "token_id_to_rvq": {},
            "num_quantizers": num_quantizers,
            "codebook_size": codebook_size,
            "compression_ratio": self._tokenizer.compression_ratio,
            "action_dim": self._tokenizer.action_dim,
        }

        all_rvq_token_ids = [rvq_group_token_id]
        for q in range(num_quantizers):
            for idx in range(codebook_size):
                token = f"<rvq_r{q}_{idx}>"
                token_id = processor.tokenizer.convert_tokens_to_ids(token)
                action_mapper["token_id_to_rvq"][token_id] = (q, idx)
                all_rvq_token_ids.append(token_id)

        action_mapper["token_range"] = {
            "min": min(all_rvq_token_ids),
            "max": max(all_rvq_token_ids),
        }

        self._action_mapper_cache = action_mapper
        return action_mapper

    def get_action_token_list(self, processor) -> List[int]:
        """v3.1 delta does not use action_token_list and returns an empty list"""
        return []

    def decode_action(
        self,
        output_ids: torch.Tensor,
        action_mapper: Dict,
        action_horizon: int,
        action_dim: int,
        device: torch.device,
        proprioception: Optional[torch.Tensor] = None,
        dof_mask: Optional[torch.Tensor] = None,
        robot_type_id: Optional[int] = None,
        state: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[Union[np.ndarray, torch.Tensor]], bool]:
        """X2Robot V3.1 Delta tokenizer decoding"""
        token_id_to_rvq = action_mapper["token_id_to_rvq"]
        num_quantizers = action_mapper["num_quantizers"]
        rvq_group_token_id = action_mapper["rvq_group_token_id"]

        # Parse RVQ tokens
        rvq_indices_list = []
        current_group = []

        for token_id_i in output_ids[0]:
            token_id = token_id_i.item()
            if token_id == rvq_group_token_id:
                if len(current_group) == num_quantizers:
                    rvq_indices_list.append(current_group)
                current_group = []
            elif token_id in token_id_to_rvq:
                q_id, idx = token_id_to_rvq[token_id]
                if len(current_group) == q_id:
                    current_group.append(idx)

        # Add the final group
        if len(current_group) == num_quantizers:
            rvq_indices_list.append(current_group)

        if len(rvq_indices_list) == 0:
            return np.zeros((action_horizon, action_dim)), False

        # Convert to tensor and decode
        indices = torch.tensor(
            rvq_indices_list, device=device
        )  # [num_latents, num_quantizers]
        indices = indices.unsqueeze(0)  # [1, num_latents, num_quantizers]

        # Get obs_state for decoding, using normalized states as in encoding
        obs_state = None
        if proprioception is not None:
            obs_state = proprioception[
                :, -1:, :
            ].float()  # [B, 1, D], convert to float32

        # Handle robot_type_id
        robot_type_id_tensor = None
        if robot_type_id is not None:
            if isinstance(robot_type_id, int):
                robot_type_id_tensor = torch.tensor(
                    [robot_type_id],
                    device=device,
                    dtype=torch.long,
                )
            elif isinstance(robot_type_id, torch.Tensor):
                robot_type_id_tensor = robot_type_id.to(device)

        predict_action = self._tokenizer.decode(
            indices,
            target_length=action_horizon,
            obs_state=obs_state,
            dof_mask=dof_mask[:, :1, :].float() if dof_mask is not None else None,
            robot_type_id=robot_type_id_tensor,
        )
        predict_action = predict_action[0]  # [T, action_dim]

        # Check whether decoding succeeded
        decode_success = False
        if isinstance(predict_action, np.ndarray):
            decode_success = np.sum(predict_action) != 0
        elif isinstance(predict_action, torch.Tensor):
            decode_success = predict_action.sum().item() != 0

        return predict_action, decode_success

    def compute_accuracy(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        action_mapper: Dict,
        action_token_id_set: Dict,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute x2robot_tokenizer_v3_1_delta accuracy

        Includes overall accuracy and per-RVQ-layer accuracy
        """
        result = {}

        shift_logits = logits[..., :-1, :].contiguous()
        action_preds = shift_logits.argmax(dim=-1)
        shift_labels = labels[..., 1:].contiguous()

        token_range = action_mapper["token_range"]
        action_mask = (shift_labels >= token_range["min"]) & (
            shift_labels <= token_range["max"]
        )
        correct_preds = (action_preds == shift_labels) & action_mask
        action_accuracy = (
            correct_preds.sum().float() / action_mask.sum().float()
            if action_mask.sum() > 0
            else torch.tensor(0.0, device=shift_labels.device)
        )
        result["action_accuracy"] = action_accuracy

        # Compute per-RVQ-layer accuracy
        num_quantizers = action_mapper["num_quantizers"]
        for q_id in range(num_quantizers):
            quantizer_mask = torch.zeros_like(action_mask)
            for token_id, (quantizer_id, _) in action_mapper["token_id_to_rvq"].items():
                if quantizer_id == q_id:
                    quantizer_mask |= shift_labels == token_id
            if quantizer_mask.sum() > 0:
                quantizer_correct = (action_preds == shift_labels) & quantizer_mask
                quantizer_accuracy = (
                    quantizer_correct.sum().float() / quantizer_mask.sum().float()
                )
                result[f"action_accuracy_rvq_q{q_id}"] = quantizer_accuracy

        return result

    def get_accuracy_keys(self) -> List[str]:
        """Return accuracy keys for x2robot_tokenizer_v3_1_delta"""
        keys = ["action_accuracy"]
        if self._tokenizer is not None:
            num_quantizers = self._tokenizer.num_quantizers
            for q_id in range(num_quantizers):
                keys.append(f"action_accuracy_rvq_q{q_id}")
        return keys

    @property
    def vocab_size(self) -> int:
        """v3.1 delta does not use vocab_size and returns 0"""
        return 0

    @property
    def uses_dof_mask_for_unnorm(self) -> bool:
        """v3.1 delta outputs full dimensions and does not need dof_mask"""
        return False

    @property
    def needs_action_crop(self) -> bool:
        """v3.1 delta does not clip actions; the encoder handles the full sequence internally"""
        return False

    def encode_to_tokens(
        self,
        actions: torch.Tensor,
        obs_state: Optional[torch.Tensor] = None,
        dof_mask: Optional[torch.Tensor] = None,
        robot_type_ids: Optional[List[int]] = None,
        is_train: bool = True,
    ) -> List[List[str]]:
        """
        X2Robot V3.1 Delta tokenizer encoding

        Args:
            actions: [B, T, D] normalized delta actions (full sequence)
            obs_state: [B, obs_horizon, D] observation frame states
            dof_mask: [B, T, D] DOF mask
            robot_type_ids: [B] robot type ID list
        """
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer not loaded. Call load_tokenizer first.")

        # Ensure tensors
        if not isinstance(actions, torch.Tensor):
            actions = torch.tensor(actions)

        # Convert NaNs to 0 so the tokenizer's apply_dof_mask can handle them correctly
        actions = actions.nan_to_num(nan=0.0)

        # Use the last obs_state frame as the reference state [B, 1, D]
        if obs_state is not None and not isinstance(obs_state, torch.Tensor):
            obs_state = torch.tensor(obs_state)
        if obs_state is not None and obs_state.dim() == 3 and obs_state.shape[1] > 1:
            obs_state = obs_state[:, -1:, :]

        # Build padding_mask (True means valid)
        B, T, _ = actions.shape
        padding_mask = torch.ones(B, T, dtype=torch.bool)

        # Handle robot_type_ids
        robot_type_ids_tensor = None
        if robot_type_ids is not None:
            if not isinstance(robot_type_ids, torch.Tensor):
                robot_type_ids_tensor = torch.tensor(robot_type_ids, dtype=torch.long)
            else:
                robot_type_ids_tensor = robot_type_ids

        # Encode
        encode_result = self._tokenizer.encode(
            actions.float(),
            obs_state=obs_state.float() if obs_state is not None else None,
            dof_mask=dof_mask.bool() if dof_mask is not None else None,
            padding_mask=padding_mask.bool(),
            robot_type_id=robot_type_ids_tensor,
        )
        indices = encode_result["indices"]  # [B, num_latents, num_quantizers]

        # Convert to token strings
        all_action_tokens = []
        B, num_latents, num_quantizers = indices.shape
        for b in range(B):
            tokens = []
            for t in range(num_latents):
                tokens.append("<rvq_group>")
                for q in range(num_quantizers):
                    idx = indices[b, t, q].item()
                    tokens.append(f"<rvq_r{q}_{idx}>")
            all_action_tokens.append(tokens)

        return all_action_tokens

    def get_robot_type_ids(
        self,
        uids: List[Optional[str]],
        dataset_names: List[str],
    ) -> Optional[List[int]]:
        """
        Get robot_type_ids for v3.1 delta

        Args:
            uids: UID list
            dataset_names: dataset name list

        Returns:
            robot_type_id list
        """
        if self._tokenizer is None:
            return None

        robot_type_ids = []
        for i in range(len(dataset_names)):
            uid = uids[i] if i < len(uids) else None
            dataset_name = dataset_names[i] if i < len(dataset_names) else None
            robot_type_id = self._tokenizer.get_robot_type_id_auto(
                uid=uid,
                dataset_name=dataset_name,
            )
            robot_type_ids.append(robot_type_id)
        return robot_type_ids

    @property
    def codebook_size(self) -> int:
        """Return the codebook size"""
        if self._tokenizer is None:
            return 0
        return self._tokenizer.codebook_size

    @property
    def num_quantizers(self) -> int:
        """Return the number of quantizers"""
        if self._tokenizer is None:
            return 0
        return self._tokenizer.num_quantizers

    @property
    def compression_ratio(self) -> int:
        """Return the compression ratio"""
        if self._tokenizer is None:
            return 0
        return self._tokenizer.compression_ratio

    @property
    def action_dim(self) -> int:
        """Return the action dimension"""
        if self._tokenizer is None:
            return 0
        return self._tokenizer.action_dim

    @property
    def robot_type_id(self) -> Optional[int]:
        """Return robot_type_id"""
        return self._robot_type_id

    @robot_type_id.setter
    def robot_type_id(self, value: Optional[int]):
        """Set robot_type_id"""
        self._robot_type_id = value

    def get_placeholder_for_dllm(self) -> List[str]:
        """Return placeholder tokens for one latent"""
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer not loaded. Call load_tokenizer first.")

        tokens = ["<rvq_placeholder_group>"]
        for q in range(self._tokenizer.num_quantizers):
            tokens.append(f"<rvq_placeholder_{q}>")
        return tokens

    @property
    def inference_ar_steps_for_dllm(self) -> int:
        """Return the AR step count (num_latents)"""
        action_horizon_ar = self.config.get("action_horizon_ar", 32)
        compression_ratio = self._tokenizer.compression_ratio
        return (action_horizon_ar + compression_ratio - 1) // compression_ratio

    def modify_inputs_for_dllm(
        self,
        inputs: Dict[str, torch.Tensor],
        processor,
        sample_time: torch.Tensor,
        dataset_names: List[str],
    ):
        """
        Prepare AR DLLM inputs by encoding sample_time into input_ids and labels

        Noise injection strategy:
        - Split [0, 1] into N equal parts (N = ar_step_num)
        - For each sample, compute the number of noisy steps k
        k = clamp(ceil((1 - t) * N), 0, N)
        -> smaller t means more noise; values closer to 1 mean less noise
        - Noise injection overwrites matching tokens in input_ids / labels step by step from seq
        """
        input_ids = inputs["input_ids"]
        labels = inputs["labels"]
        prefix_length = inputs["prefix_length"]
        bs, seqlen = input_ids.shape

        # v3.1 delta parameters
        action_horizon_ar = self.config.get("action_horizon_ar", 32)
        compression_ratio = self._tokenizer.compression_ratio
        ar_step_num = (action_horizon_ar + compression_ratio - 1) // compression_ratio
        step_token_len = 1 + self._tokenizer.num_quantizers
        ar_token_length = ar_step_num * step_token_len

        device = input_ids.device
        dtype = input_ids.dtype

        placeholder_ids = torch.tensor(
            processor.placeholder_seq, device=device, dtype=dtype
        )
        noise_seq = placeholder_ids.repeat(ar_step_num)

        if not torch.is_tensor(sample_time):
            sample_time = torch.tensor(sample_time, device=device, dtype=torch.float32)
        else:
            sample_time = sample_time.to(device=device, dtype=torch.float32)

        sample_time = sample_time.clamp(0.0, 1.0)

        noisy_steps_per_sample = (
            torch.ceil((1.0 - sample_time) * (ar_step_num + 1)).long() - 1
        )
        noisy_steps_per_sample = noisy_steps_per_sample.clamp(min=0, max=ar_step_num)

        start = prefix_length - ar_token_length - 2
        end = prefix_length - 2

        action_idx = 0  # action sample counter
        for i in range(bs):
            if not is_action_dataset_name(dataset_names[i]):
                continue
            k = noisy_steps_per_sample[action_idx].item()
            action_idx += 1
            perm = torch.randperm(ar_step_num, device=device)
            chosen_steps = perm[:k]

            step_mask = torch.zeros(ar_step_num, dtype=torch.bool, device=device)
            step_mask[chosen_steps] = True

            token_mask = step_mask.repeat_interleave(step_token_len)
            ignore_mask = ~token_mask

            cur_input_view = input_ids[i, start:end]
            cur_label_view = labels[i, start + 1 : end + 1]

            cur_input_view[token_mask] = noise_seq[token_mask]
            cur_label_view[ignore_mask] = -100

        inputs["input_ids"] = input_ids
        inputs["labels"] = labels
        return inputs

    def update_positional_masks_for_dllm(
        self, positional_masks, inputs, processor, visible_predict_ar_ratio=1
    ):
        """Update DLLM positional masks"""
        if self.dllm and self.input_placeholder_flag:
            mask = self.update_placeholder_mask(
                processor,
                inputs["prefix_length"],
                inputs["input_ids"],
            )
            positional_masks.update(mask)
            if np.random.rand() < visible_predict_ar_ratio:
                positional_masks["ar_visible"] = True
                if "ar_predict_token_positions" in positional_masks:
                    del positional_masks["ar_predict_token_positions"]
            else:
                positional_masks["ar_visible"] = False
        return positional_masks

    def update_placeholder_mask(self, processor, prefix_length, input_ids):
        """Generate ar_action_mask"""
        ar_action_mask = torch.zeros_like(input_ids)

        action_horizon_ar = self.config.get("action_horizon_ar", 32)
        compression_ratio = self._tokenizer.compression_ratio
        num_latents = (action_horizon_ar + compression_ratio - 1) // compression_ratio
        step_len = 1 + self._tokenizer.num_quantizers

        for b, seq in enumerate(input_ids):
            current_index = 1
            for i in range(num_latents):
                start_idx = prefix_length - (num_latents - i) * step_len - 2
                end_idx = start_idx + step_len
                ar_action_mask[b, start_idx:end_idx] = current_index
                current_index += 1
            ar_action_mask[b, end_idx:prefix_length] = -1  # eos

        return {"ar_action_mask": ar_action_mask}

    def init_inference(self, robot_type: Optional[str] = None) -> None:
        """
        Initialize inference and set robot_type_id

        Args:
            robot_type: robot type name
        """
        if robot_type is not None and self._tokenizer is not None:
            self._robot_type_id = self._tokenizer.get_robot_type_id_auto(
                robot_type=robot_type,
                robot_path=robot_type,
                dataset_name=robot_type,
            )

    def get_robot_type_id_auto(
        self,
        robot_type: str = None,
        robot_path: str = None,
        dataset_name: str = None,
    ) -> Optional[int]:
        """
        Resolve robot_type_id automatically

        Args:
            robot_type: robot type name
            robot_path: robot path
            dataset_name: dataset name

        Returns:
            robot_type_id, or None if it cannot be resolved
        """
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer not loaded. Call load_tokenizer first.")

        robot_type_id = self._tokenizer.get_robot_type_id_auto(
            robot_type=robot_type,
            robot_path=robot_path,
            dataset_name=dataset_name,
        )
        self._robot_type_id = robot_type_id
        return robot_type_id


# ============================================================
# Factory function
# ============================================================

_TOKENIZER_REGISTRY: Dict[str, type] = {
    "fast": FastTokenizerMixin,
    "spatialvla": SpatialVLATokenizerMixin,
    "x2robot_tokenizer_v3_1_delta": X2RobotV31DeltaTokenizerMixin,
}


def get_action_tokenizer_mixin(tokenizer_type: str) -> ActionTokenizerMixin:
    """
    Get the mixin instance for a tokenizer type

    Args:
        tokenizer_type: tokenizer type, supports "fast", "spatialvla", "x2robot_tokenizer_v3_1_delta"

    Returns:
        ActionTokenizerMixin instance

    Raises:
        ValueError: Unsupported tokenizer type
    """
    if tokenizer_type not in _TOKENIZER_REGISTRY:
        raise ValueError(
            f"Unsupported action tokenizer type: {tokenizer_type}. "
            f"Supported types: {list(_TOKENIZER_REGISTRY.keys())}"
        )
    return _TOKENIZER_REGISTRY[tokenizer_type]()
