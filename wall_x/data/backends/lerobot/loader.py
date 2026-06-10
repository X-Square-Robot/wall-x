"""
LeRobot Dataset Loader - Distributed Version
"""

import logging
import os
from typing import Protocol, SupportsIndex, TypeVar

import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from qwen_vl_utils.vision_process import smart_resize
from torch.utils.data import DistributedSampler, random_split
from transformers import AutoProcessor

from wall_x._vendor.x2robot_utils.geometry import (
    canonicalize_euler_zyx_batch_nb,
    euler_to_matrix_zyx_batch_nb,
    matrix_to_euler_zyx_batch_nb,
    so3_to_matrix_batch_nb,
)
from wall_x.data.backends.lerobot.config import LerobotConfig
from wall_x.data.backends.lerobot.rotation_layout import (
    LAYOUT_SKIP_KEYS,
    maybe_convert_euler_to_6d,
)
from wall_x.data.backends.lerobot.rotation_layout import (
    euler_layout_dim as _euler_layout_dim,
)
from wall_x.data.backends.lerobot.rotation_layout import (
    layout_uses_6d_rotation as _layout_uses_6d_rotation,
)
from wall_x.data.backends.lerobot.utils import (
    get_wallx_normal_text,
    load_norm_stats,
    preprocesser_call,
    process_grounding_points,
    replace_action_token,
)

T_co = TypeVar("T_co", covariant=True)
logger = logging.getLogger(__name__)

RELATIVE_KEYWORD = "relative"
ROTATION_KEYWORD = "rotation"
RELATIVE_SKIP_KEYS = LAYOUT_SKIP_KEYS


def _compute_delta_from_state_and_abs_rot(
    rotations: np.ndarray, state: np.ndarray
) -> np.ndarray:
    """Relative rotation: R_rel = R_abs @ R_state^T."""
    if rotations.shape[-1] == 3:
        rotations_matrix = euler_to_matrix_zyx_batch_nb(rotations)
        out_is_euler = True
    elif rotations.shape[-1] == 6:
        rotations_matrix = so3_to_matrix_batch_nb(rotations)
        out_is_euler = False
    else:
        raise ValueError(
            f"Only 3D euler or 6D rotation supported, got {rotations.shape[-1]}D"
        )

    if state.shape[-1] == 3:
        state_matrix = euler_to_matrix_zyx_batch_nb(state[np.newaxis, :])[0]
    elif state.shape[-1] == 6:
        state_matrix = so3_to_matrix_batch_nb(state[np.newaxis, :])[0]
    else:
        raise ValueError(
            f"Only 3D euler or 6D rotation supported, got {state.shape[-1]}D"
        )

    r_rel = np.matmul(rotations_matrix, state_matrix.T)
    if out_is_euler:
        d_euler = matrix_to_euler_zyx_batch_nb(r_rel)
        return canonicalize_euler_zyx_batch_nb(d_euler)
    return r_rel[:, :2, :].reshape(r_rel.shape[0], 6)


# Abstract class for dataset
class Dataset(Protocol[T_co]):
    """Interface for a dataset with random access."""

    def __getitem__(self, index: SupportsIndex) -> T_co:
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class PreprocessedDataset(Dataset[T_co]):
    def __init__(
        self,
        dataset,
        config,
        norm_stats,
        dataload_config,
        lerobot_config,
        seed=42,
        rank=0,
        world_size=1,
        test_only=False,
    ):
        self.hf_dataset = dataset

        if test_only:
            self._dataset = dataset
        else:
            self._dataset = None
            self.train_dataset, self.val_dataset = random_split(
                dataset,
                [0.95, 0.05],
                torch.Generator().manual_seed(seed) if seed is not None else None,
            )
            self._train()

        self.seed = seed
        self.rank = rank
        self.world_size = world_size

        # init configs
        self.config = config
        self.use_fast_tokenizer = self.config.get("use_fast_tokenizer", False)
        self.dataload_config = dataload_config
        self.norm_stats = norm_stats
        self.lerobot_config = lerobot_config

        self.data_config = LerobotConfig().update(
            train_test_split=self.dataload_config["train_test_split"],
            seed=self.dataload_config["seed"],
            resolution=self.dataload_config.get("resolution", None),
            priority_order=self.dataload_config.get("priority_order", None),
            camera_name_mapping=self.dataload_config.get("camera_name_mapping", None),
        )

        self.key_mappings = self.dataload_config["key_mappings"]

        self._cam_key_mapping = self.key_mappings["camera"]
        self._state_key_mapping = self.key_mappings
        self._action_key_mapping = self.key_mappings

        task_cfg = self.config.get("task") or {}
        self._dof_config = self.config.get("dof_config") or task_cfg.get(
            "dof_config", {}
        )
        self._agent_pos_config = self.config.get("agent_pos_config") or task_cfg.get(
            "agent_pos_config", {}
        )
        self._use_relative_action = any(
            RELATIVE_KEYWORD in key for key in self._dof_config
        )
        self._convert_action_euler_to_6d = _layout_uses_6d_rotation(self._dof_config)
        self._convert_state_euler_to_6d = _layout_uses_6d_rotation(
            self._agent_pos_config
        )
        if self._convert_action_euler_to_6d or self._convert_state_euler_to_6d:
            logger.info(
                "LeRobot loader: Euler->6D rotation enabled "
                "(action=%s, state=%s; raw action dim=%s -> %s)",
                self._convert_action_euler_to_6d,
                self._convert_state_euler_to_6d,
                (
                    _euler_layout_dim(self._dof_config)
                    if self._convert_action_euler_to_6d
                    else "-"
                ),
                (
                    sum(
                        d
                        for k, d in self._dof_config.items()
                        if k not in RELATIVE_SKIP_KEYS
                    )
                    if self._convert_action_euler_to_6d
                    else "-"
                ),
            )

    def _maybe_convert_euler_to_6d(self, vec, layout_config: dict, enabled: bool):
        converted = maybe_convert_euler_to_6d(vec, layout_config, enabled)
        if (
            enabled
            and layout_config
            and isinstance(vec, torch.Tensor)
            and converted is not vec
        ):
            return torch.as_tensor(converted, dtype=vec.dtype, device=vec.device)
        return converted

    def _to_relative_action(self, action, agent_pos):
        """Convert absolute action horizon to deltas w.r.t. current agent_pos."""
        action = np.asarray(action, dtype=np.float64)
        agent_pos = np.asarray(agent_pos, dtype=np.float64)
        if action.ndim == 1:
            action = action[np.newaxis, :]
        if agent_pos.ndim > 1:
            agent_pos = agent_pos.reshape(-1)

        parts = []
        cur = 0
        for key, dim in self._dof_config.items():
            if key in RELATIVE_SKIP_KEYS:
                continue
            action_clip = action[:, cur : cur + dim]
            agent_pos_clip = agent_pos[cur : cur + dim]
            if RELATIVE_KEYWORD not in key:
                parts.append(action_clip)
            elif ROTATION_KEYWORD in key:
                parts.append(
                    _compute_delta_from_state_and_abs_rot(
                        action_clip.astype(np.float64),
                        agent_pos_clip.astype(np.float64),
                    )
                )
            else:
                parts.append(action_clip - agent_pos_clip[np.newaxis, :])
            cur += dim

        if not parts:
            return action
        return np.concatenate(parts, axis=1).astype(np.float32)

    def _vision_preprocess(self, frames):
        processed_frames = []
        for key in self.hf_dataset.meta.camera_keys:
            from PIL import Image

            current_obs = frames[key].clone().permute(1, 2, 0)

            img_pil = Image.fromarray((current_obs * 255).to(torch.uint8).cpu().numpy())
            orig_width, orig_height = img_pil.size
            # 2. Apply resolution constraints (if config is not -1)
            target_size = self.data_config.resolution.get(
                self._cam_key_mapping[key], -1
            )
            if target_size != -1:
                # Maintain aspect ratio logic
                if orig_width > orig_height:  # Landscape image
                    new_width = target_size
                    new_height = int(target_size * orig_height / orig_width)
                else:  # Portrait image
                    new_height = target_size
                    new_width = int(target_size * orig_width / orig_height)
                img_pil = img_pil.resize((new_width, new_height))

            # 3. Apply smart scaling (qwen logic)
            current_width, current_height = img_pil.size
            resized_height, resized_width = smart_resize(
                current_height,
                current_width,
                factor=self.data_config.image_factor,
                min_pixels=self.data_config.min_pixels,
                max_pixels=self.data_config.max_pixels,
            )
            resized_img = img_pil.resize((resized_width, resized_height))
            processed_frames.append(resized_img)

        return processed_frames, orig_height, orig_width, resized_height, resized_width

    def __getitem__(self, index):
        data = self._dataset[index]
        image_inputs, h, w, resize_h, resize_w = self._vision_preprocess(data)
        agent_pos = data[self._state_key_mapping["state"]]
        action = data[self._action_key_mapping["action"]]
        agent_pos = self._maybe_convert_euler_to_6d(
            agent_pos, self._agent_pos_config, self._convert_state_euler_to_6d
        )
        action = self._maybe_convert_euler_to_6d(
            action, self._dof_config, self._convert_action_euler_to_6d
        )
        if self._use_relative_action:
            device = action.device if isinstance(action, torch.Tensor) else None
            action = torch.as_tensor(
                self._to_relative_action(action, agent_pos),
                dtype=torch.float32,
                device=device,
            )
        frame_index = data["frame_index"]
        instruction_info = {"instruction": data["task"]}
        generate_subtask_ratio = self.data_config.generate_subtask_ratio
        complete_text, generate_subtask = get_wallx_normal_text(
            instruction_info,
            self.dataload_config.get("action_horizon", 33) - 1,
            frame_index,
            self.data_config.priority_order,
            self._cam_key_mapping,
            generate_subtask_ratio=generate_subtask_ratio,
            camera_name_mapping=self.data_config.camera_name_mapping,
        )
        text = process_grounding_points(
            complete_text, h, w, resize_h, resize_w, self.data_config.model_type
        )
        result = {
            "image_inputs": image_inputs,
            "text": text,
            "action": action,
            "agent_pos": agent_pos,
            "frame_index": frame_index,
        }

        return result

    def __len__(self) -> int:
        return len(self._dataset)

    def _eval(self):
        self._dataset = self.val_dataset

    def _train(self):
        self._dataset = self.train_dataset

    def get_train_dataloader(self):
        """
        Get distributed training dataloader

        Args:
            rank: Current process rank
            world_size: Total number of processes
            seed: Random seed for reproducibility
        """
        self._train()

        batch_size = self.config.get("batch_size_per_gpu", 8)
        num_workers = self.config.get("num_workers", 4)

        # Create distributed sampler
        sampler = DistributedSampler(
            self,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=True,
            seed=self.seed,
            drop_last=True,  # Ensure all processes have same number of batches
        )

        dataloader = torch.utils.data.DataLoader(
            self,
            batch_size=batch_size,
            sampler=sampler,  # Use distributed sampler instead of shuffle=True
            num_workers=num_workers,
            collate_fn=DataCollator(
                self.config, self.dataload_config, self.norm_stats, self.lerobot_config
            ),
            pin_memory=True,  # Enable for GPU training
            persistent_workers=num_workers > 0,  # Only if num_workers > 0
            prefetch_factor=2,  # Reduce memory usage
            drop_last=True,  # Avoid incomplete batches
        )

        return dataloader, sampler

    def get_val_dataloader(self):
        """
        Get distributed evaluation dataloader (no shuffling for consistent evaluation)
        """
        self._eval()

        batch_size = self.config.get(
            "eval_batch_size_per_gpu", self.config.get("batch_size_per_gpu", 8)
        )
        num_workers = self.config.get("num_workers", 4)

        # Create distributed sampler for evaluation (no shuffle)
        sampler = DistributedSampler(
            self,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=False,  # No shuffling for evaluation
            drop_last=False,  # Keep all samples for evaluation
        )

        dataloader = torch.utils.data.DataLoader(
            self,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            collate_fn=DataCollator(
                self.config, self.dataload_config, self.norm_stats, self.lerobot_config
            ),
            pin_memory=True,
            persistent_workers=num_workers > 0,
            prefetch_factor=2,
            drop_last=False,
        )

        return dataloader, sampler


class DataCollator:
    # Class-level cache for processors to avoid reloading
    _processor_cache = {}
    _action_tokenizer_cache = {}
    _norm_stat_alignment_warnings = set()

    def __init__(self, config, dataload_config, stats, lerobot_config):
        self.config = config
        self.dataload_config = dataload_config
        self.stats = stats
        self.action_min_stat = stats["action"].min
        self.action_delta = stats["action"].delta
        self.state_min_stat = stats["state"].min
        self.state_delta = stats["state"].delta
        self.lerobot_config = lerobot_config
        self.np_rng = np.random.default_rng()

        noise_scheduler_config = config.get("noise_scheduler", {})
        self.beta_alpha = noise_scheduler_config.get(
            "beta_alpha", 1.5
        )  # alpha parameter of the Beta distribution
        self.beta_beta = noise_scheduler_config.get(
            "beta_beta", 1.0
        )  # beta parameter of the Beta distribution
        self.s = noise_scheduler_config.get("s", 0.999)  # scaling factor
        self.time_shift = noise_scheduler_config.get(
            "time_shift", 1.0
        )  # time shift factor

        self.beta_alpha = float(self.beta_alpha)
        self.beta_beta = float(self.beta_beta)
        self.use_fast_tokenizer = self.config.get("use_fast_tokenizer", False)
        self.use_state_string_representation = bool(
            self.config.get("use_state_string_representation", False)
        )
        self.state_bins = int(self.config.get("state_bins", 256))
        self.load_processor()

    def load_processor(self):
        processor_path = self.config["processor_path"]
        action_tokenizer_path = self.config.get("action_tokenizer_path", None)

        if (
            self.use_fast_tokenizer
            and action_tokenizer_path not in self._action_tokenizer_cache
        ):
            self._action_tokenizer_cache[action_tokenizer_path] = (
                AutoProcessor.from_pretrained(
                    action_tokenizer_path, trust_remote_code=True
                )
            )

        # Use cached processors if available
        if processor_path not in self._processor_cache:
            processor = AutoProcessor.from_pretrained(processor_path, use_fast=True)
            if self.config.get("padding_side", "left") == "left":
                processor.tokenizer.padding_side = "left"

            new_tokens = ["<|propri|>", "<|action|>"]
            processor.tokenizer.add_tokens(new_tokens)
            if self.use_fast_tokenizer and self.config.get("model_type") == "qwen2_5":
                action_tokenizer = self._action_tokenizer_cache[action_tokenizer_path]
                new_tokens = [
                    f"<|action_token_{i}|>" for i in range(action_tokenizer.vocab_size)
                ]
                processor.tokenizer.add_tokens(new_tokens)
                begin_idx_token = "<|action_token_0|>"
                token_id = processor.tokenizer.convert_tokens_to_ids(begin_idx_token)
                processor.tokenizer.init_kwargs["action_token_start_index"] = token_id
                processor.tokenizer.init_kwargs["action_token_vocab_size"] = (
                    action_tokenizer.vocab_size
                )

            self._processor_cache[processor_path] = processor

        self.processor = self._processor_cache[processor_path]

        if not self.use_fast_tokenizer:
            self.train_action_tokenizer = None
        else:
            self.train_action_tokenizer = self._action_tokenizer_cache[
                action_tokenizer_path
            ]

    @classmethod
    def _normalize(cls, action, min_stat, delta):
        """
        Normalize action data using min-max normalization.
        """
        delta = torch.where(delta == 0, torch.ones_like(delta), delta)
        x = (action - min_stat) / delta
        x = x * 2 - 1
        x = torch.clamp(x, -1, 1)
        return x

    @staticmethod
    def _align_norm_stat(stat, value, *, pad_value: float, name: str):
        """Align a 1-D norm stat with the current LeRobot tensor width."""
        stat = stat.to(device=value.device, dtype=value.dtype)
        target_dim = value.shape[-1]
        stat_dim = stat.shape[-1]
        if stat_dim == target_dim:
            return stat
        if stat_dim > target_dim:
            warning_key = ("truncate", name, stat_dim, target_dim)
            if warning_key not in DataCollator._norm_stat_alignment_warnings:
                logger.warning(
                    "Truncating LeRobot %s norm stat from %s to %s dims",
                    name,
                    stat_dim,
                    target_dim,
                )
                DataCollator._norm_stat_alignment_warnings.add(warning_key)
            return stat[..., :target_dim]
        pad_shape = (*stat.shape[:-1], target_dim - stat_dim)
        pad = torch.full(pad_shape, pad_value, device=value.device, dtype=value.dtype)
        warning_key = ("pad", name, stat_dim, target_dim)
        if warning_key not in DataCollator._norm_stat_alignment_warnings:
            logger.warning(
                "Padding LeRobot %s norm stat from %s to %s dims",
                name,
                stat_dim,
                target_dim,
            )
            DataCollator._norm_stat_alignment_warnings.add(warning_key)
        return torch.cat([stat, pad], dim=-1)

    def __call__(self, batch):
        additional_inputs = {}

        # Tail-pad widths when dof_config / agent_pos_config (sum) is larger
        # than the lerobot action/state - typical when resuming a ckpt that
        # was pretrained on a bigger action space. Extra columns are filled
        # with zeros and their mask set to 0 so loss is not propagated.
        dof_total = int(self.config.get("dof_total_dim", 0) or 0)
        agent_pos_total = int(self.config.get("agent_pos_total_dim", 0) or 0)

        # Explicit init so the ``if action is not None`` guard and later
        # ``replace_action_token`` call stay well-defined even if a batch
        # unexpectedly omits the action / agent_pos keys. Without this the
        # loop-local variables would leak NameError on the first miss.
        action = None
        dof_mask = None
        agent_pos = None
        agent_pos_mask = None

        for key in batch[0].keys():
            if key == "agent_pos":
                agent_pos = torch.stack([item["agent_pos"] for item in batch])
                if agent_pos.dim() == 2:
                    agent_pos = agent_pos.unsqueeze(1)
                agent_pos_mask = (~torch.isnan(agent_pos)).float()
                agent_pos.nan_to_num_(nan=0.0)
                state_min_stat = self._align_norm_stat(
                    self.state_min_stat,
                    agent_pos,
                    pad_value=0.0,
                    name="state.min",
                )
                state_delta = self._align_norm_stat(
                    self.state_delta,
                    agent_pos,
                    pad_value=1.0,
                    name="state.delta",
                )
                agent_pos = self._normalize(agent_pos, state_min_stat, state_delta)
                if agent_pos_total and agent_pos.shape[-1] < agent_pos_total:
                    pad_w = agent_pos_total - agent_pos.shape[-1]
                    agent_pos = torch.nn.functional.pad(agent_pos, (0, pad_w))
                    agent_pos_mask = torch.nn.functional.pad(agent_pos_mask, (0, pad_w))
                additional_inputs["proprioception"] = agent_pos
                additional_inputs["agent_pos_mask"] = agent_pos_mask
            elif key == "action":
                action = torch.stack([item["action"] for item in batch])
                if action.dim() == 2:
                    action = action.unsqueeze(1)
                dof_mask = (~torch.isnan(action)).float()
                action.nan_to_num_(nan=0.0)
                action_min_stat = self._align_norm_stat(
                    self.action_min_stat,
                    action,
                    pad_value=0.0,
                    name="action.min",
                )
                action_delta = self._align_norm_stat(
                    self.action_delta,
                    action,
                    pad_value=1.0,
                    name="action.delta",
                )
                action = self._normalize(action, action_min_stat, action_delta)
                if dof_total and action.shape[-1] < dof_total:
                    pad_w = dof_total - action.shape[-1]
                    action = torch.nn.functional.pad(action, (0, pad_w))
                    dof_mask = torch.nn.functional.pad(dof_mask, (0, pad_w))
                additional_inputs["action_chunk"] = action
                additional_inputs["dof_mask"] = dof_mask
            elif key == "image_inputs":
                additional_inputs["image_inputs"] = [
                    item["image_inputs"] for item in batch
                ]
            elif key == "text":
                additional_inputs["text"] = [item["text"] for item in batch]
            elif key == "frame_index":
                additional_inputs["frame_index"] = torch.stack(
                    [item["frame_index"] for item in batch]
                )
            else:
                raise NotImplementedError(
                    f"{key} input not implemented in preprocesser"
                )

        # sample noise time
        if action is not None:
            sample_time = self.sample_time(
                action.shape[0],
                device=action.device,
                dtype=torch.float32,
            )
            additional_inputs["sample_time"] = sample_time

        additional_inputs["text"] = replace_action_token(
            additional_inputs["text"],
            additional_inputs["action_chunk"],
            self.train_action_tokenizer if self.use_fast_tokenizer else None,
            additional_inputs["dof_mask"],
        )

        inputs = preprocesser_call(
            processor=self.processor,
            text=additional_inputs.pop("text"),
            images=additional_inputs.pop("image_inputs"),
            videos=None,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self.dataload_config.get("max_length", 768),
            norm_state=(
                additional_inputs["proprioception"]
                if self.use_state_string_representation
                and "proprioception" in additional_inputs
                else None
            ),
            agent_pos_mask=additional_inputs.get("agent_pos_mask"),
            state_bins=self.state_bins,
        )

        action_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|action|>")

        # Gating token types
        additional_inputs["moe_token_types"] = inputs.input_ids == action_token_id

        inputs.update(additional_inputs)

        inputs["dataset_names"] = [self.lerobot_config["repo_id"]] * inputs[
            "action_chunk"
        ].shape[0]

        return inputs

    def sample_time(self, batch_size, device, dtype):
        """
        Sample timesteps

        Use a Beta distribution to sample values in [0, 1], then scale them.

        Args:
            batch_size (int): batch size
            device: Device type
            dtype: dtype

        Returns:
            torch.Tensor: sampled timesteps with shape [batch_size]
        """

        sample_np = self.np_rng.beta(
            self.beta_alpha, self.beta_beta, size=(batch_size,)
        ).astype(np.float32)
        sample = torch.from_numpy(sample_np).to(
            device=device, dtype=dtype, non_blocking=True
        )

        # sample = self.beta_dist.sample([batch_size]).to(dtype=dtype)
        time = 1 - sample

        # Apply diffusion time shift
        if self.time_shift != 1.0:
            time = (self.time_shift * time) / (1 + (self.time_shift - 1) * time)

        time = time * self.s  # noise should denoise from 0 to 1 here
        return time


def load_lerobot_data(
    config,
    lerobot_config,
    rank=0,
    world_size=1,
    seed=42,
):
    """
    Load LeRobot dataset with distributed support

    Args:
        config: Model configuration
        rank: Current process rank (default: 0)
        world_size: Total number of processes (default: 1)
        seed: Random seed for reproducibility (default: 42)

    Returns:
        dataset: Training dataset
        train_num: Number of training samples per process
        sampler: Distributed sampler (None if world_size=1)
    """

    # Set seed for reproducibility
    torch.manual_seed(seed)

    dataload_config = get_data_configs(config["data"])
    key_mappings = dataload_config["key_mappings"]

    repo_id = lerobot_config.get("repo_id", None)
    assert repo_id is not None, "repo id is required"
    root = lerobot_config.get("root", None)
    meta_info = LeRobotDatasetMetadata(repo_id, root=root)
    dataset_fps = meta_info.fps
    episodes_num = meta_info.total_episodes

    norm_stats_path = config.get("norm_stats_path", None)
    assert (
        norm_stats_path is not None
    ), "norm stats is required, please refer to 'wall-x/scripts/compute_norm_stats.py' to compute stats"
    task_cfg = config.get("task") or {}
    dof_config = config.get("dof_config") or task_cfg.get("dof_config", {})
    agent_pos_config = config.get("agent_pos_config") or task_cfg.get(
        "agent_pos_config", {}
    )
    norm_stats = load_norm_stats(
        norm_stats_path,
        key_mappings,
        dof_config=dof_config,
        agent_pos_config=agent_pos_config,
    )

    delta_timestamps = {
        # action chunk
        key_mappings["action"]: [
            t / dataset_fps
            for t in range(dataload_config.get("action_horizon", 33) - 1)
        ],
    }
    batch_size = config.get("batch_size_per_gpu", 8)

    # Optional episode subset. YAML ``lerobot_config.episodes`` has always
    # been present in examples but previously ignored; honour it so smoke
    # tests / small-dataset runs don't pay the O(N) LeRobotDataset indexing
    # cost on a multi-thousand-episode repo (~10s / episode on some formats).
    episodes_override = lerobot_config.get("episodes")
    if episodes_override is not None:
        episodes = list(episodes_override)
        episodes_num_effective = len(episodes)
    else:
        episodes = np.arange(episodes_num).tolist()
        episodes_num_effective = episodes_num

    train_test_split = dataload_config.get("train_test_split", 0.95)
    split_idx = int(episodes_num_effective * train_test_split)
    # Guard: tiny episode subsets + high train_test_split can floor split_idx
    # to 0 (e.g. 1 ep * 0.95 = 0), which would silently hand LeRobotDataset an
    # empty episode list and end training after 0 iterations. Fail loud.
    if split_idx < 1:
        raise ValueError(
            f"train_test_split={train_test_split} applied to "
            f"{episodes_num_effective} episode(s) yields 0 train episodes. "
            f"Use more episodes or a lower train_test_split."
        )
    train_episodes = episodes[:split_idx]
    test_episodes = episodes[split_idx:]

    global_rank = torch.distributed.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
    # TODO: Some LeRobot formats need to load all metadata before splitting
    # episodes; loading from all ranks at once can exhaust memory.
    train_dataset = None

    # Sequential loading inside each node
    for r in range(local_world_size):
        if local_rank == r:
            logger.info(
                "[Global rank %s] Loading dataset on local_rank=%s",
                global_rank,
                local_rank,
            )

            train_dataset = LeRobotDataset(
                repo_id=repo_id,
                root=root,
                episodes=train_episodes,
                delta_timestamps=delta_timestamps,
                video_backend="pyav",
            )

            logger.info(
                "[Global rank %s] Finished loading on local_rank=%s",
                global_rank,
                local_rank,
            )

        # Barrier only within the node
        torch.distributed.barrier(device_ids=[local_rank])

    if rank == 0:
        logger.info("Selected train episodes: %s", train_dataset.episodes)
        logger.info("Number of train episodes selected: %s", train_dataset.num_episodes)
        logger.info("Number of train frames selected: %s", train_dataset.num_frames)
        logger.info("Selected test episodes: %s", test_episodes)

    dataset = PreprocessedDataset(
        train_dataset,
        config,
        norm_stats,
        dataload_config,
        lerobot_config,
        seed=seed,
        rank=rank,
        world_size=world_size,
    )

    # Calculate samples per process
    if world_size > 1:
        # With DistributedSampler, each process gets approximately len(dataset) // world_size samples
        samples_per_process = len(dataset) // world_size
        train_num = samples_per_process // batch_size
    else:
        train_num = len(dataset) // batch_size

    if rank == 0:
        lines = [
            "LeRobot Data Loading Configuration:",
            f"  rank: {rank}",
            f"  world_size: {world_size}",
            f"  batch_size_per_gpu: {batch_size}",
            f"  repo_id: {repo_id}",
            f"  total_dataset_size: {len(dataset)}",
        ]
        if world_size > 1:
            lines.extend(
                [
                    f"  samples_per_process: {samples_per_process}",
                    f"  batches_per_process: {train_num}",
                    f"  total_batches_all_processes: {train_num * world_size}",
                ]
            )
        else:
            lines.append(f"  total_batches: {train_num}")
        lines.append(f"  seed: {seed}")
        logger.info("\n%s", "\n".join(lines))

    return dataset, train_num


def get_distributed_dataloader(
    dataset, config, rank=0, world_size=1, seed=42, is_train=True
):
    """
    Helper function to get distributed dataloader

    Args:
        dataset: PreprocessedDataset instance
        config: Configuration dict
        rank: Current process rank
        world_size: Total number of processes
        seed: Random seed
        is_train: Whether this is for training (affects shuffling)

    Returns:
        dataloader: Distributed DataLoader
        sampler: DistributedSampler
    """
    if is_train:
        return dataset.get_train_dataloader(rank=rank, world_size=world_size, seed=seed)
    else:
        return dataset.get_val_dataloader(rank=rank, world_size=world_size)


def get_data_configs(config):
    default_data_config = {
        "train_test_split": 0.95,
        "seed": 42,
        "batch_size": 8,
        "action_horizon": 21,
        "action_history_length": 0,
        "image_horizon": 1,
        "image_history_length": 0,
        "left_padding": False,
        "right_padding": False,
        "return_first_obs": False,
        "return_last_obs": False,
        "randomize_obs_after": None,
        "datasets": [],
        "labeled_pathes": [],
        "camera_name_mapping": None,
    }
    data_config = default_data_config | config
    data_config["action_horizon"] += 1

    return data_config
