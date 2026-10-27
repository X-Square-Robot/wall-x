"""Object-box crop of memory frames (evaluation convention of training)."""

from __future__ import annotations

from typing import Tuple

import numpy as np
from PIL import Image

from wall_x.serving.keyframe_memory.grammar import BOX_SCALE
from wall_x.serving.keyframe_memory.state import Box

EXPAND = 0.20  # each edge moves outward by this fraction of the box side
MIN_SIDE = 56  # px at the processing resolution; smaller crops grow around the centre


def crop_rect(box: Box, width: int, height: int) -> Tuple[int, int, int, int]:
    """Integer crop rectangle for a 0..1000 ``box`` on a ``width`` x ``height`` image."""
    sx, sy = width / BOX_SCALE, height / BOX_SCALE
    x1, y1, x2, y2 = box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy
    bw, bh = x2 - x1, y2 - y1
    x1, x2 = x1 - EXPAND * bw, x2 + EXPAND * bw
    y1, y2 = y1 - EXPAND * bh, y2 + EXPAND * bh

    def clamp(lo: float, hi: float, limit: int) -> Tuple[float, float]:
        lo, hi = max(0.0, lo), min(float(limit), hi)
        side = min(float(MIN_SIDE), float(limit))
        if hi - lo < side:
            centre = (lo + hi) / 2.0
            lo, hi = centre - side / 2.0, centre + side / 2.0
            if lo < 0.0:
                hi, lo = hi - lo, 0.0
            if hi > limit:
                lo, hi = lo - (hi - limit), float(limit)
            lo = max(0.0, lo)
        return lo, hi

    x1, x2 = clamp(x1, x2, width)
    y1, y2 = clamp(y1, y2, height)
    ix2, iy2 = min(int(round(x2)), width), min(int(round(y2)), height)
    ix1 = max(0, min(int(round(x1)), ix2 - 1))
    iy1 = max(0, min(int(round(y1)), iy2 - 1))
    return ix1, iy1, ix2, iy2


def crop_to_box(image, box: Box):
    """Crop an already-resized memory image to its box and resize it back.

    Keeping the size unchanged keeps the image-token count of the prompt
    unchanged. Accepts and returns a PIL image or an HWC uint8 array.
    """
    pil = image if isinstance(image, Image.Image) else Image.fromarray(np.asarray(image))
    width, height = pil.size
    out = pil.crop(crop_rect(box, width, height)).resize((width, height), Image.BICUBIC)
    return out if isinstance(image, Image.Image) else np.asarray(out)
