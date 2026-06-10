"""Grounding-point helpers (point / bbox coordinate remap, pure regex)."""

import re
from typing import List, Optional


def process_grounding_points(
    text: str, orig_height, orig_width, resized_height, resized_width, model_type
) -> str:
    """Remap <point>/<box>/<bbox> coordinates inside ``text`` from the original
    image size to the resized space used by the given model type.
    """
    point_pattern = re.compile(r"<(point|box|bbox)>(.*?)</\1>")

    def process_match(match):
        tag_name = match.group(1)
        coords_str = match.group(2)
        try:
            coords = list(map(int, re.findall(r"\d+", coords_str)))

            scale_w = resized_width / orig_width
            scale_h = resized_height / orig_height

            if len(coords) == 2:
                x, y = coords
                if model_type == "qwen2_5":
                    new_x = max(0, min(round(x * scale_w), resized_width - 1))
                    new_y = max(0, min(round(y * scale_h), resized_height - 1))
                elif model_type in ["qwen2"]:
                    new_x = max(0, min(999.999, (x / orig_width) * 1000))
                    new_y = max(0, min(999.999, (y / orig_height) * 1000))
                else:
                    raise ValueError("Unsupported model type")
                coords = [new_x, new_y]

            if len(coords) == 4:
                x1, y1, x2, y2 = coords
                if model_type == "qwen2_5":
                    new_x1 = max(0, min(round(x1 * scale_w), resized_width - 1))
                    new_y1 = max(0, min(round(y1 * scale_h), resized_height - 1))
                    new_x2 = max(0, min(round(x2 * scale_w), resized_width - 1))
                    new_y2 = max(0, min(round(y2 * scale_h), resized_height - 1))
                elif model_type in ["qwen2"]:
                    new_x1 = max(0, min(999.999, (x1 / orig_width) * 1000))
                    new_y1 = max(0, min(999.999, (y1 / orig_height) * 1000))
                    new_x2 = max(0, min(999.999, (x2 / orig_width) * 1000))
                    new_y2 = max(0, min(999.999, (y2 / orig_height) * 1000))
                else:
                    raise ValueError("Unsupported model type")
                coords = [new_x1, new_y1, new_x2, new_y2]

            return f'<{tag_name}>[{", ".join(map(str, coords))}]</{tag_name}>'

        except (ValueError, TypeError):
            return match.group(0)

    return point_pattern.sub(process_match, text)


def extract_grounding_points(text: str) -> List[List[float]]:
    """Extract all <point>/<box>/<bbox> coordinates from ``text`` as list-of-list."""
    point_pattern = re.compile(r"<(point|box|bbox)>\s*\[([^\]]+)\]\s*</\1>")

    points: List[List[float]] = []
    for match in point_pattern.finditer(text):
        coords_str = match.group(2)
        raw_values = re.findall(r"-?\d+\.?\d*", coords_str)
        converted: List[float] = []
        for value in raw_values:
            number = float(value)
            converted.append(int(number) if number.is_integer() else number)
        if converted:
            points.append(converted)

    return points


def reverse_grounding_points(
    text: str, orig_height, orig_width, resized_height, resized_width, model_type
) -> str:
    """Inverse of ``process_grounding_points`` - map resized coords back to original."""
    point_pattern = re.compile(r"<(point|box|bbox)>(.*?)</\1>")

    def reverse_match(match):
        tag_name = match.group(1)
        coords_str = match.group(2)
        try:
            coords = list(map(float, re.findall(r"-?\d+\.?\d*", coords_str)))

            scale_w = resized_width / orig_width
            scale_h = resized_height / orig_height

            if len(coords) == 2:
                x, y = coords
                if model_type == "qwen2_5":
                    orig_x = max(0, min(orig_width - 1, round(x / scale_w)))
                    orig_y = max(0, min(orig_height - 1, round(y / scale_h)))
                elif model_type in ["qwen2"]:
                    orig_x = max(
                        0, min(orig_width - 1, round((x / 1000) * orig_width))
                    )
                    orig_y = max(
                        0, min(orig_height - 1, round((y / 1000) * orig_height))
                    )
                else:
                    raise ValueError("Unsupported model type")
                coords = [orig_x, orig_y]

            if len(coords) == 4:
                x1, y1, x2, y2 = coords
                if model_type == "qwen2_5":
                    orig_x1 = max(0, min(orig_width - 1, round(x1 / scale_w)))
                    orig_y1 = max(0, min(orig_height - 1, round(y1 / scale_h)))
                    orig_x2 = max(0, min(orig_width - 1, round(x2 / scale_w)))
                    orig_y2 = max(0, min(orig_height - 1, round(y2 / scale_h)))
                elif model_type in ["qwen2"]:
                    orig_x1 = max(
                        0, min(orig_width - 1, round((x1 / 1000) * orig_width))
                    )
                    orig_y1 = max(
                        0, min(orig_height - 1, round((y1 / 1000) * orig_height))
                    )
                    orig_x2 = max(
                        0, min(orig_width - 1, round((x2 / 1000) * orig_width))
                    )
                    orig_y2 = max(
                        0, min(orig_height - 1, round((y2 / 1000) * orig_height))
                    )
                else:
                    raise ValueError("Unsupported model type")
                coords = [orig_x1, orig_y1, orig_x2, orig_y2]

            return f'<{tag_name}>[{", ".join(map(str, map(int, coords)))}]</{tag_name}>'

        except (ValueError, TypeError):
            return match.group(0)

    return point_pattern.sub(reverse_match, text)


def calculate_point_l1_distance(gt_text: str, pred_text: str) -> Optional[float]:
    """Average L1 distance between 2D points extracted from ``gt_text`` / ``pred_text``.

    Returns None if either side has no points or counts differ.
    """
    point_pattern = re.compile(r"<point>\[(\d+),\s*(\d+)\]</point>")

    gt_matches = point_pattern.findall(gt_text)
    pred_matches = point_pattern.findall(pred_text)

    if not gt_matches or not pred_matches or len(gt_matches) != len(pred_matches):
        return None

    total_l1_distance = 0.0
    for (gt_x, gt_y), (pred_x, pred_y) in zip(gt_matches, pred_matches):
        try:
            gt_x, gt_y = int(gt_x), int(gt_y)
            pred_x, pred_y = int(pred_x), int(pred_y)
            l1_dist = abs(gt_x - pred_x) + abs(gt_y - pred_y)
            total_l1_distance += l1_dist
        except ValueError:
            continue

    return total_l1_distance / len(gt_matches) if gt_matches else None


__all__ = [
    "process_grounding_points",
    "extract_grounding_points",
    "reverse_grounding_points",
    "calculate_point_l1_distance",
]
