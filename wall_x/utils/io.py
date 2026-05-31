"""Model I/O utilities — canonical load/save for state dicts and videos."""

import os
import hashlib
import torch
import numpy as np
import imageio
from tqdm import tqdm
from contextlib import contextmanager
from safetensors.torch import load_file as load_safetensors


def load_state_dict(file_path, torch_dtype=None, device="cpu"):
    """Load model weights from file(s).

    Supports .safetensors, .bin, .pt, .pth formats. Accepts a single path
    or a list of paths (weights are merged).

    Args:
        file_path: Weight file path or list of paths.
        torch_dtype: Target data type (only converts float types).
        device: Target device.

    Returns:
        State dict mapping parameter names to tensors.
    """
    if isinstance(file_path, list):
        state_dict = {}
        for path in file_path:
            state_dict.update(
                load_state_dict(path, torch_dtype=torch_dtype, device=device)
            )
        return state_dict

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Model file not found: {file_path}")

    if file_path.endswith(".safetensors"):
        state_dict = load_safetensors(file_path, device=str(device))
    else:
        state_dict = torch.load(file_path, map_location=device, weights_only=True)

    if torch_dtype is not None:
        state_dict = {
            k: (
                v.to(torch_dtype)
                if v.dtype in [torch.float32, torch.float16, torch.bfloat16]
                else v
            )
            for k, v in state_dict.items()
        }

    return state_dict


def load_state_dict_from_folder(folder_path, torch_dtype=None, device="cpu"):
    """Load and merge state dicts from all model files in a folder."""
    state_dict = {}
    for file_name in sorted(os.listdir(folder_path)):
        ext = file_name.rsplit(".", 1)[-1] if "." in file_name else ""
        if ext in ("safetensors", "bin", "ckpt", "pth", "pt"):
            state_dict.update(
                load_state_dict(
                    os.path.join(folder_path, file_name),
                    torch_dtype=torch_dtype,
                    device=device,
                )
            )
    return state_dict


def save_video(frames, save_path, fps, quality=9, ffmpeg_params=None):
    """Save frames as video file using imageio."""
    writer = imageio.get_writer(
        save_path, fps=fps, quality=quality, ffmpeg_params=ffmpeg_params
    )
    for frame in tqdm(frames, desc="Saving video", leave=False):
        frame = np.array(frame)
        writer.append_data(frame)
    writer.close()


def convert_state_dict_keys_to_single_str(state_dict, with_shape=True):
    """Convert state_dict keys to a single string for hashing."""
    keys = []
    for key, value in state_dict.items():
        if isinstance(key, str):
            if isinstance(value, torch.Tensor):
                if with_shape:
                    shape = "_".join(map(str, list(value.shape)))
                    keys.append(key + ":" + shape)
                keys.append(key)
            elif isinstance(value, dict):
                keys.append(
                    key
                    + "|"
                    + convert_state_dict_keys_to_single_str(
                        value, with_shape=with_shape
                    )
                )
    keys.sort()
    return ",".join(keys)


def hash_state_dict_keys(state_dict, with_shape=True):
    """Calculate MD5 hash of state_dict keys to identify model configuration."""
    keys_str = convert_state_dict_keys_to_single_str(state_dict, with_shape=with_shape)
    return hashlib.md5(keys_str.encode("UTF-8")).hexdigest()


@contextmanager
def init_weights_on_device(device=torch.device("meta"), include_buffers=False):
    """Context manager to initialize model parameters on specified device."""
    old_register_parameter = torch.nn.Module.register_parameter
    if include_buffers:
        old_register_buffer = torch.nn.Module.register_buffer

    def register_empty_parameter(module, name, param):
        old_register_parameter(module, name, param)
        if param is not None:
            param_cls = type(module._parameters[name])
            kwargs = module._parameters[name].__dict__
            kwargs["requires_grad"] = param.requires_grad
            module._parameters[name] = param_cls(
                module._parameters[name].to(device), **kwargs
            )

    def register_empty_buffer(module, name, buffer, persistent=True):
        old_register_buffer(module, name, buffer, persistent=persistent)
        if buffer is not None:
            module._buffers[name] = module._buffers[name].to(device)

    def patch_tensor_constructor(fn):
        def wrapper(*args, **kwargs):
            kwargs["device"] = device
            return fn(*args, **kwargs)

        return wrapper

    if include_buffers:
        tensor_constructors_to_patch = {
            name: getattr(torch, name) for name in ["empty", "zeros", "ones", "full"]
        }
    else:
        tensor_constructors_to_patch = {}

    try:
        torch.nn.Module.register_parameter = register_empty_parameter
        if include_buffers:
            torch.nn.Module.register_buffer = register_empty_buffer
        for name in tensor_constructors_to_patch:
            setattr(torch, name, patch_tensor_constructor(getattr(torch, name)))
        yield
    finally:
        torch.nn.Module.register_parameter = old_register_parameter
        if include_buffers:
            torch.nn.Module.register_buffer = old_register_buffer
        for name, old_fn in tensor_constructors_to_patch.items():
            setattr(torch, name, old_fn)


def print_model_info(model, name="Model"):
    """Print model parameter counts."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"{name}: {total_params/1e9:.2f}B params, {trainable_params/1e9:.2f}B trainable"
    )
