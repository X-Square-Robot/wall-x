from typing import Dict, List
import logging
import numpy as np
from wall_x.data.utils import preprocesser_call
from qwen_vl_utils.vision_process import smart_resize
import torch
from PIL import Image
from transformers import BatchFeature

logger = logging.getLogger(__name__)


def prepare_batch(
    obs: Dict,
    processor,
    normalizer_propri,
    camera_key: List[str],
    agent_pos_dim,
    action_dim,
    pred_horizon,
    fixed_action_dim,
    max_length,
    image_factor: int,
    min_pixels: int,
    max_pixels: int,
    predict_mode: str = "fast",
    device: str = "cuda",
) -> BatchFeature:
    """Prepare observation into model input format.

    Args:
        obs: Dictionary containing:
            - 'camera_key_0' : image 0
            - 'camera_key_1' : image 1
            ...
            - 'prompt': Text prompt
            - 'state': Robot state/proprioception
            - 'dataset_names': Dataset names

    Returns:
        BatchFeature object ready for model input
    """
    # Handle images - can be single image, list of images, or dict of images
    images = []
    images = [obs[key] for key in camera_key]
    # Convert numpy arrays to PIL Images
    processed_images = []
    for img in images:
        if isinstance(img, np.ndarray):
            # Debug: Log the shape and dtype
            logger.debug(f"Image shape: {img.shape}, dtype: {img.dtype}")

            # Handle unexpected dimensions - squeeze if needed
            if img.ndim > 3:
                logger.warning(
                    f"Image has {img.ndim} dimensions, squeezing extra dimensions"
                )
                img = np.squeeze(img)

            # Verify shape is valid for PIL
            if img.ndim == 2:
                # Grayscale image
                pass
            elif img.ndim == 3:
                # Check if channel dimension is first or last
                if img.shape[0] == 3 or img.shape[0] == 1:
                    # Channels first, transpose to channels last
                    img = np.transpose(img, (1, 2, 0))
                elif img.shape[2] == 3 or img.shape[2] == 1:
                    # Already channels last
                    pass
                else:
                    raise ValueError(
                        f"Unexpected image shape: {img.shape}. Expected (H, W, C) or (C, H, W)"
                    )
            else:
                raise ValueError(
                    f"Invalid image dimensions: {img.ndim}. Expected 2 or 3 dimensions, got shape {img.shape}"
                )

            # Convert to PIL Image
            if img.dtype == np.uint8:
                img = Image.fromarray(img)
            else:
                img = Image.fromarray((img * 255).astype(np.uint8))
        processed_images.append(img)

    # print("processed_images:",processed_images)
    # Apply smart resize to images
    resized_images = process_images(
        processed_images, image_factor, min_pixels, max_pixels
    )

    # Handle text prompt - format with vision tokens
    instruction = obs["prompt"]
    formatted_text = format_text_with_vision_tokens(
        instruction, camera_key, predict_mode, pred_horizon
    )

    # Use processor to prepare inputs
    inputs = preprocesser_call(
        processor=processor,
        text=[formatted_text],
        images=[resized_images],
        videos=None,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=max_length,
    )

    action_token_id = processor.tokenizer.convert_tokens_to_ids("<|action|>")
    moe_token_types = inputs.input_ids == action_token_id
    inputs["moe_token_types"] = torch.tensor(moe_token_types)

    # obs["dataset_names"]="libero_all"

    # Handle robot state/proprioception if available
    if "state" in obs:
        state = obs["state"]
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float()
        elif not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)

        # Add batch dimension if needed
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if state.dim() == 2:
            state = state.unsqueeze(1)  # [batch, 1, state_dim]

        # Pad to 20 dimensions if needed (same as training)
        # if state.shape[-1] < 20:
        #     padding = torch.zeros(state.shape[0], state.shape[1], 20 - state.shape[-1])
        #     state = torch.cat([state, padding], dim=-1)

        # Create mask for valid dimensions
        agent_pos_mask = torch.ones_like(state)
        if state.shape[-1] > agent_pos_dim:
            agent_pos_mask[:, :, agent_pos_dim:] = 0

        normalizer_propri.normalize_data(state, [obs["dataset_names"]]*state.shape[0])
        
        inputs["proprioception"] = state
        inputs["agent_pos_mask"] = agent_pos_mask

    # Add dataset name (required by model)
    inputs["dataset_names"] = [obs["dataset_names"]]*state.shape[0]

    # Move all tensors to device
    for key in inputs:
        if isinstance(inputs[key], torch.Tensor):
            inputs[key] = inputs[key].to(device)

    dof_mask = torch.ones([state.shape[0], pred_horizon, fixed_action_dim])
    dof_mask[:, :, action_dim:] = 0

    inputs["dof_mask"] = dof_mask

    # Convert to BatchFeature to maintain consistency with training pipeline
    return BatchFeature(data=dict(inputs)).to(device)


def process_images(
    images: List[Image.Image], image_factor: int, min_pixels: int, max_pixels: int
) -> List[Image.Image]:
    """Process images with smart resize following the data loading pattern.

    Args:
        images: List of PIL Images

    Returns:
        List of resized PIL Images
    """
    resized_images = []
    for img_pil in images:
        
        orig_width, orig_height = img_pil.size
        target_size = 256
        if target_size != -1:
            # Maintain aspect ratio logic
            if orig_width > orig_height:  # Landscape image
                new_width = target_size
                new_height = int(target_size * orig_height / orig_width)
            else:  # Portrait image
                new_height = target_size
                new_width = int(target_size * orig_width / orig_height)
            img_pil = img_pil.resize((new_width, new_height))

        # Apply smart scaling (Qwen logic)
        current_width, current_height = img_pil.size
        resized_height, resized_width = smart_resize(
            current_height,
            current_width,
            factor=image_factor,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )

        resized_img = img_pil.resize((resized_width, resized_height))
        resized_images.append(resized_img)

    return resized_images

def format_text_with_vision_tokens(
    instruction: str,
    camera_key: List[str],
    predict_mode: str = "diffusion",
    pred_horizon: int = 32,
) -> str:
    """Format text prompt with vision tokens for the model.

    Args:
        instruction: Task instruction text
        camera_key: List of camera names

    Returns:
        Formatted text with special tokens
    """
    # Special tokens for formatting
    role_start_symbol = "<|im_start|>"
    role_end_symbol = "<|im_end|>"
    vision_start_symbol = "<|vision_start|>"
    vision_end_symbol = "<|vision_end|>"
    image_pad_symbol = "<|image_pad|>"
    propri_symbol = "<|propri|>"
    action_symbol = "<|action|>"
    action_fast_symbol = "<|action_fast|>"

    # Camera name mapping
    camera_name_mapping = {
        "front_view": "front view",
        "face_view": "front view",
        "left_wrist_view": "left wrist view",
        "right_wrist_view": "right wrist view",
        "top_view": "top view",
        "wall_view": "wall view",
    }

    # System prologue
    prologue = (
        f"{role_start_symbol}system\nYou are a helpful assistant.{role_end_symbol}\n"
    )

    # User request with observation
    user_request = f"{role_start_symbol}user\nObservation:"
    if camera_key:
        for cam_name in camera_key:
            view_name = camera_name_mapping.get(cam_name, cam_name)
            user_request += f" {view_name}: {vision_start_symbol}{image_pad_symbol}{vision_end_symbol}"
    user_request += "\nInstruction:"

    text_prompt = (
        f"\nPredict the next action in robot action.\nProprioception: {propri_symbol}\n"
    )
    user_message = f"{user_request} {instruction}{text_prompt}{role_end_symbol}\n"
    assistant_output = f"{role_start_symbol}assistant\n{action_fast_symbol}{role_end_symbol}\n"
    if predict_mode == "diffusion":
        assistant_output = f"{role_start_symbol}assistant\n{action_symbol * pred_horizon}{role_end_symbol}\n"
    complete_text = prologue + user_message + assistant_output

    return complete_text
