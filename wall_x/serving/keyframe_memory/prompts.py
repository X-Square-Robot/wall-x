"""Prompt text of the two passes, byte-for-byte as rendered in training."""

from __future__ import annotations

from typing import Optional

IMAGE = "<|vision_start|><|image_pad|><|vision_end|>"
SELECTION_SYSTEM_PROMPT = "You are a helpful assistant."
SELECTION_QUESTION = (
    "Which recent frames record newly confirmed task progress "
    "that must be kept in long-term memory?"
)


def selection_prompt(instruction: str, num_memory: int, num_candidates: int) -> str:
    """Selection-pass prompt, ending where the decision token is generated.

    The assistant turn opens with an empty think block, as in training, so
    decoding starts at ``<|kf_write|>`` / ``<|kf_skip|>``.
    """
    lines = [f"Task: {instruction}"]
    if num_memory:
        lines.append(
            "Progress memory: "
            + " ".join(f"[M_{j}]: {IMAGE}" for j in range(num_memory))
        )
    else:
        lines.append("Progress memory: (empty)")
    lines.append(
        "Recent frames: "
        + " ".join(f"<|R_{i}|>: {IMAGE}" for i in range(num_candidates))
    )
    user = ("\n".join(lines) + "\n" + SELECTION_QUESTION).strip()
    return (
        f"<|im_start|>system\n{SELECTION_SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )


def inject_memory(
    prefix: str,
    *,
    num_memory: int,
    num_short_term: int,
    frame: Optional[int],
    label: str,
    memory_in_prompt: bool = True,
) -> str:
    """Insert memory blocks after ``Observation:`` of an action prompt.

    The images they reference are prepended to the camera images, in the same
    order: memory, then short-term taps, then the current cameras.
    """
    if memory_in_prompt and (num_memory or num_short_term):
        parts = []
        if num_memory:
            parts.append(f" {label}:")
            parts.extend(f" [M_{j}]: {IMAGE}" for j in range(num_memory))
        if num_short_term:
            parts.append(" Recent view:")
            parts.extend(f" {IMAGE}" for _ in range(num_short_term))
        parts.append(f" now[f{frame}]:" if frame is not None else " now:")
        return prefix.replace("Observation:", "Observation:" + "".join(parts), 1)
    if frame is not None:
        return prefix.replace("Observation:", f"Observation: now[f{frame}]:", 1)
    return prefix
