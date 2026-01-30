import os
import yaml
import torch
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from wall_x.model.qwen2_5_based.modeling_qwen2_5_vl_act import Qwen2_5_VLMoEForAction
from wall_x.data.load_lerobot_dataset import load_test_dataset, get_data_configs
from wall_x.model.model_utils import register_normalizers
import copy

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    config["data"]["model_type"] = config.get("model_type")

    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_horizon", type=int, default=32)
    parser.add_argument("--origin_action_dim", type=int, default=14)
    args = parser.parse_args()

    origin_action_dim = args.origin_action_dim
    pred_horizon = args.pred_horizon

    # get train config
    model_path = "/path/to/your/checkpoint"
    action_tokenizer_path = "/path/to/Models/fast"
    save_dir = f"/path/to/save/dir"
    path = f"{model_path}/config.yml"
    config = load_config(path)

    normalizer_action, normalizer_propri = register_normalizers(config,model_path)

    # load model with customized robot config
    model = Qwen2_5_VLMoEForAction.from_pretrained(
        model_path, train_config=config, action_tokenizer_path=action_tokenizer_path
    )
    
    model.set_normalizer(copy.deepcopy(normalizer_action), copy.deepcopy(normalizer_propri))
    model.eval()
    model = model.to("cuda")
    model.to_bfloat16_for_selected_params()


    # get test dataloader
    dataload_config = get_data_configs(config["data"])
    lerobot_config = dataload_config.get("lerobot_config", {})
    dataset = load_test_dataset(config, lerobot_config, normalizer_action, normalizer_propri, seed=42)
    dataloader = dataset.get_dataloader()
    # dataloader = dataset.get_train_dataloader()

    total_frames = len(dataloader)

    predict_mode = "fast" if config.get("use_fast_tokenizer", False) else "diffusion"
    action_dim = 14 if predict_mode == "diffusion" else origin_action_dim
    gt_traj = torch.zeros((total_frames, origin_action_dim))
    pred_traj = torch.zeros((total_frames, origin_action_dim))

    # use tqdm to show the progress
    for idx, batch in tqdm(
        enumerate(dataloader), total=total_frames, desc="predicting"
    ):
        if idx % pred_horizon == 0 and idx + pred_horizon < total_frames:
            batch = batch.to("cuda")
            with torch.no_grad():
                outputs = model(
                    **batch,
                    action_dim=action_dim,
                    action_horizon=pred_horizon,
                    mode="predict",
                    predict_mode=predict_mode,
                )
                pred_traj[idx : idx + pred_horizon] = (
                    outputs["predict_action"][:, :, :origin_action_dim]
                    .detach()
                    .cpu()
                    .squeeze(0)
                )

            # Denormalize ground truth actions
            gt_action_chunk = batch["action_chunk"][:, :, :origin_action_dim]
            dof_mask = batch["dof_mask"].to(gt_action_chunk.dtype)
            denormalized_gt = (
                model.action_preprocessor.normalizer_action.unnormalize_data(
                    gt_action_chunk,
                    [lerobot_config.get("repo_id", "physical-intelligence/libero")],
                    dof_mask,
                ).squeeze(0)
            )
            gt_traj[idx : idx + pred_horizon] = denormalized_gt.detach().cpu()

    gt_traj_np = gt_traj.numpy()
    pred_traj_np = pred_traj.numpy()

    timesteps = gt_traj.shape[0]

    fig, axs = plt.subplots(
        origin_action_dim, 1, figsize=(15, 5 * origin_action_dim), sharex=True
    )
    fig.suptitle("Action Comparison for lerobot", fontsize=16)

    for i in range(origin_action_dim):
        axs[i].plot(range(timesteps), gt_traj_np[:, i], label="Ground Truth")
        axs[i].plot(range(timesteps), pred_traj_np[:, i], label="Prediction")
        axs[i].set_ylabel(f"Action Dim {i+1}")
        axs[i].legend()
        axs[i].grid(True)

    axs[-1].set_xlabel("Timestep")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "lerobot_comparison.png")
    plt.savefig(save_path)
    print(f"Saved plot to {save_path}")
    plt.close()
