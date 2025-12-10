import yaml
import torch
import tqdm
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from wall_x.data.load_lerobot_dataset import KEY_MAPPINGS
import normalize
import numpy as np
import argparse


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    config["data"]["model_type"] = config.get("model_type")

    return config


def load_lerobot_dataset(repo_id, root, action_horizon, args):
    dataset_meta = LeRobotDatasetMetadata(repo_id)
    dataset = LeRobotDataset(
        repo_id,
        root=root,
        delta_timestamps={
            key: [t / dataset_meta.fps for t in range(action_horizon)]
            for key in [KEY_MAPPINGS[repo_id]["action"]]
        },
        video_backend="pyav",
    )
    num_batches = len(dataset) // args.batch_size
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=True,
        generator=generator,
        num_workers=args.num_workers,
        persistent_workers=True if args.num_workers > 0 else False,
    )
    return data_loader, num_batches


if __name__ == "__main__":
    # set args
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # Configs
    path = "/path/to/config.yml"
    output_path = "/path/to/output"
    config = load_config(path)
    lerobot_config = config["data"]["lerobot_config"]
    repo_id = lerobot_config.get("repo_id", None)
    root = lerobot_config.get("root", None)
    assert repo_id is not None, "repo id is required"
    action_horizon = config["data"].get("action_horizon", 32)

    data_loader, num_batches = load_lerobot_dataset(repo_id, root, action_horizon, args)

    keys = ["state", "action"]
    stats = {key: normalize.RunningStats() for key in keys}
    for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Computing stats"):
        for key in keys:
            stats[key].update(np.asarray(batch[KEY_MAPPINGS[repo_id][key]]))
    norm_stats = {
        KEY_MAPPINGS[repo_id][key]: stats.get_statistics()
        for key, stats in stats.items()
    }

    output_path = output_path + "/" + repo_id
    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)
