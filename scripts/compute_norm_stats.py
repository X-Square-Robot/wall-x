#!/usr/bin/env python3

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterator, List, Tuple
from tqdm import tqdm

import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset



def write_json(path: Path, data: Dict) -> None:
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

def compute_action_statistics(
    action_data_by_robot: Dict[str, Dict[str, List]]
) -> Dict[str, Dict[str, Dict]]:
    """
    Compute statistics (min, q01, q99, max) for each action type and dimension.

    Args:
        action_data_by_robot: Dict[robot_id][action_type] -> list of arrays/lists

    Returns:
        Dict[robot_id][action_type] -> {
            "min": [min for each dim],
            "q01": [quantile 1% for each dim],
            "q99": [quantile 99% for each dim],
            "max": [max for each dim],
            "delta": [max - min for each dim]
            "delta_q99_q01": [q99 - q01 for each dim]
        }
    """
    stats = {}

    for robot_id, action_data in action_data_by_robot.items():
        stats[robot_id] = {}

        for action_type, values_list in action_data.items():
            if not values_list:
                continue

            # Convert to numpy array: shape (num_samples, num_dims)
            try:
                values_array = np.array(values_list)
                if values_array.size == 0:
                    continue

                # Handle both 1D and 2D cases
                if values_array.ndim == 1:
                    values_array = values_array.reshape(-1, 1)
                elif values_array.ndim == 2:
                    pass
                else:
                    logging.warning(
                        f"Unexpected shape for {robot_id}/{action_type}: {values_array.shape}"
                    )
                    continue

                # Compute statistics for each dimension
                min_vals = np.min(values_array, axis=0).tolist()
                max_vals = np.max(values_array, axis=0).tolist()
                q01_vals = np.quantile(values_array, 0.01, axis=0).tolist()
                q99_vals = np.quantile(values_array, 0.99, axis=0).tolist()
                delta_vals = (np.array(max_vals) - np.array(min_vals)).tolist()
                delta_q99_q01_vals = (np.array(q99_vals) - np.array(q01_vals)).tolist()

                stats[robot_id][action_type] = {
                    "min": min_vals,
                    "q01": q01_vals,
                    "q99": q99_vals,
                    "max": max_vals,
                    "delta": delta_vals,
                    "delta_q99_q01": delta_q99_q01_vals,
                }

            except Exception as e:
                logging.warning(
                    f"Error computing statistics for {robot_id}/{action_type}: {e}"
                )
                continue

    return stats


def load_lerobot_dataset(
    repo_id: str,
    trajectory_keys: Dict,
    base_dir: Path,
) -> None:
    
    # 加载本地或远程数据集
    dataset = LeRobotDataset(base_dir)

    # 遍历所有数据
    frames: Dict[str, Dict[str, List]] = defaultdict(
        lambda: defaultdict(list)
    )

    all_features = dataset.features
    non_image_columns = [col for col in all_features if "image" not in col]

    print(f"正在读取以下字段: {non_image_columns}")
    fast_dataset = dataset.hf_dataset.select_columns(non_image_columns)

    for i in tqdm(range(len(fast_dataset))):
        sample = fast_dataset[i]
        action = sample["action"]             # torch.Tensor
        propri  = sample["observation.state"] 

        for key, action_keys in trajectory_keys.items():
            for action_key, action_range in action_keys.items():
                if  key == "action":
                    frames[repo_id][action_key].append(action[action_range[0]:action_range[1]].numpy().tolist())
                else:
                    frames[repo_id][action_key].append(propri[action_range[0]:action_range[1]].numpy().tolist())

    return frames



def compute_action_normalizer(repo_id: str, trajectory_keys: Dict, base_dir: Path, output_dir: Path) -> None:
    """
    Compute action normalizer statistics for all robot_ids.
    """
    logging.info("Starting action normalizer computation...")

    # Collect action data by robot_id
    action_data_by_robot: Dict[str, Dict[str, List]] = defaultdict(
        lambda: defaultdict(list)
    )

    frames = load_lerobot_dataset(repo_id,trajectory_keys, base_dir)
    
    # Compute statistics
    stats = compute_action_statistics(frames)

    # Save statistics for each robot_id
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # for robot_id, robot_stats in stats.items():
    #     output_file = output_dir / f"{robot_id}_action_stats.json"
    #     write_json(output_file, robot_stats)
    #     logging.info(f"Saved action statistics for {robot_id} to {output_file}")

    # Also save a combined file
    combined_output = output_dir / "all_robots_action_stats.json"
    write_json(combined_output, stats)
    logging.info(f"Saved combined action statistics to {combined_output}")


def main() -> None:

    repo_id = "xxx" # your dataset name
    data_root_path = "/path/to/lerobot/dataset"
    output_stats_dir = "/path/to/save/action_stats"
    trajectory_keys = {                             # your dataset keys
        "action":{
            "follow_right_ee_cartesian_pos": [0,3],
            "follow_right_ee_rotation": [3,6],
            "follow_right_gripper": [6,7]
        },
        "propri":{
            "master_right_ee_cartesian_pos": [0,3],
            "master_right_ee_rotation": [3,6],
            "master_right_gripper": [6,7]
        }
    }

    compute_action_normalizer(repo_id, trajectory_keys, data_root_path, output_stats_dir)
    logging.info("Action normalizer computation completed.")


if __name__ == "__main__":
    main()
