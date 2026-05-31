import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import torch
import re
import os
from PIL import Image
from scipy.spatial.transform import Rotation


def visualize_batch(batch, future_frame_interval=4, iteration=0):
    batch = dict(batch)
    batch_size = len(batch["uids"])
    num_images = [item.count("<|vision_start|>") for item in batch["prefix_text"]]
    start_indices = [0] + [sum(num_images[:i]) for i in range(1, len(num_images))]
    end_indices = start_indices[1:] + [len(batch["image_inputs"])]
    batch["image_inputs"] = [
        batch["image_inputs"][start_indices[i] : end_indices[i]]
        for i in range(len(start_indices))
    ]

    action_chunk = np.zeros((batch_size, *batch["action_chunk"].shape[1:]))
    action_chunk[: batch["action_chunk"].shape[0]] = (
        batch["action_chunk"].float().cpu().numpy()
    )
    batch["action_chunk"] = action_chunk

    proprioception = np.zeros((batch_size, *batch["proprioception"].shape[1:]))
    proprioception[: batch["proprioception"].shape[0]] = (
        batch["proprioception"].float().cpu().numpy()
    )
    batch["proprioception"] = proprioception

    image_future = []
    for i in range(batch_size):
        if 3 * i < len(batch["image_future"]):
            image_future.append(batch["image_future"][3 * i : 3 * (i + 1)])
        else:
            image_future.append(None)
    batch["image_future"] = image_future

    for i in range(batch_size):
        sample = get_sample_from_batch(batch, i, batch_size=batch_size)
        create_comprehensive_gif(
            sample,
            interval=future_frame_interval,
            save_path=f"vis/dataset_sample_comprehensive_{iteration}_{i}.gif",
            fps=2,
        )


def get_sample_from_batch(batch, idx, batch_size=8):
    if isinstance(batch, torch.Tensor) or isinstance(batch, np.ndarray):
        if batch.shape[0] == batch_size:
            return batch[idx]
        else:
            return None
    elif isinstance(batch, dict):
        return {k: get_sample_from_batch(v, idx, batch_size) for k, v in batch.items()}
    elif isinstance(batch, list) and len(batch) == batch_size:
        return batch[idx]
    elif isinstance(batch, list):
        if len(batch) % batch_size == 0:
            length = len(batch) // batch_size
            return batch[idx * length : (idx + 1) * length]
        else:
            return None
    elif batch is None:
        return None
    else:
        raise ValueError(f"Unknown type: {type(batch)}")


def compress_image_pads(text, core_text="image_pad"):
    """Collapse consecutive <|core_text|> tokens into <|core_text|>*N format."""
    pattern = rf"(?:<\|{core_text}\|>)+"

    def replace_func(match):
        count = match.group(0).count(f"<|{core_text}|>")
        return f"<|{core_text}|>*{count}"

    return re.sub(pattern, replace_func, text)


def create_comprehensive_gif(
    frame_data, interval=4, save_path="comprehensive_animation.gif", fps=2
):
    """Create a comprehensive animated GIF showing videos, actions, and metadata."""
    video_inputs = frame_data["image_inputs"]
    video_future = frame_data["image_future"]

    if isinstance(video_inputs, Image.Image):
        video_inputs = [video_inputs]
    if video_future is None:
        video_future = [[] for _ in video_inputs]
    videos = [
        [v_input] + v_future for v_input, v_future in zip(video_inputs, video_future)
    ]
    instruction_text = frame_data["prefix_text"]
    if "postfix_text" in frame_data:
        instruction_text += frame_data["postfix_text"]

    if (
        "Observation:" in instruction_text
        and "<|vision_start|>" in instruction_text
        and "<|vision_end|>" in instruction_text
    ):
        visual_part = instruction_text.split("Observation:")[1].split("Instruction:")[0]
        camera_names = [
            t.split("<|vision_start|>")[0].strip().replace(":", "")
            for t in visual_part.split("<|vision_end|>")
        ][:-1]
    else:
        camera_names = ["Front View", "Left Wrist View", "Right Wrist View"]

    instruction_text = compress_image_pads(instruction_text, core_text="image_pad")
    instruction_text = compress_image_pads(instruction_text, core_text="action")

    max_frames = 0
    processed_videos = []
    for video in videos:
        video_np = video.numpy() if isinstance(video, torch.Tensor) else np.array(video)
        processed_videos.append(video_np)
        max_frames = max(max_frames, video_np.shape[0])

    fig = plt.figure(figsize=(20, 20))
    gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)

    video_axes: list[plt.Axes] = []
    video_ims: list[matplotlib.image.AxesImage] = []
    for i, camera_name in enumerate(camera_names):
        ax = fig.add_subplot(gs[0, i])
        ax.set_title(camera_name, fontsize=12)
        ax.axis("off")
        im = ax.imshow(np.zeros((252, 196, 3)), aspect="auto")
        video_axes.append(ax)
        video_ims.append(im)

    ax_actions = fig.add_subplot(gs[0, 3])
    action = frame_data["action_chunk"]
    time_steps = np.arange(action.shape[0])
    colors = plt.cm.tab10(np.linspace(0, 1, min(7, action.shape[1])))
    for i in range(min(7, action.shape[1])):
        ax_actions.plot(
            time_steps,
            action[:, i],
            color=colors[i],
            linewidth=1.5,
            label=f"DOF {i}",
            alpha=0.6,
        )
    ax_actions.set_title("Action Sequences", fontsize=12)
    ax_actions.set_xlabel("Time Steps")
    ax_actions.set_ylabel("Action Value")
    ax_actions.legend(fontsize=8)
    ax_actions.grid(True, alpha=0.3)
    frame_line_1 = ax_actions.axvline(x=0, color="red", linewidth=3, alpha=0.8)

    ax_actions = fig.add_subplot(gs[1, 3])
    colors = plt.cm.tab10(np.linspace(0, 1, min(7, action.shape[1] - 7)))
    for i in range(min(7, action.shape[1] - 7)):
        ax_actions.plot(
            time_steps,
            action[:, i + 7],
            color=colors[i],
            linewidth=1.5,
            label=f"DOF {i + 7}",
            alpha=0.6,
        )
    ax_actions.set_title("Action Sequences", fontsize=12)
    ax_actions.set_xlabel("Time Steps")
    ax_actions.set_ylabel("Action Value")
    ax_actions.legend(fontsize=8)
    ax_actions.grid(True, alpha=0.3)
    frame_line_2 = ax_actions.axvline(x=0, color="red", linewidth=3, alpha=0.8)

    ax_agent = fig.add_subplot(gs[1, 0])
    agent_pos = frame_data["proprioception"].flatten()
    bars = ax_agent.bar(range(len(agent_pos)), agent_pos, color="skyblue", alpha=0.7)
    ax_agent.set_title("Agent Position", fontsize=12)
    ax_agent.set_xlabel("DOF Index")
    ax_agent.set_ylabel("Position Value")
    ax_agent.grid(True, alpha=0.3)
    for bar, val in zip(bars, agent_pos):
        ax_agent.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    axis_len = 0.2
    all_left_positions, all_right_positions = [], []
    for t in range(action.shape[0]):
        if action.shape[1] >= 14:
            all_left_positions.append(action[t, 0:3])
            all_right_positions.append(action[t, 7:10])

    def _setup_3d_ax(gs_pos, title, positions, colors, labels):
        ax = fig.add_subplot(gs_pos, projection="3d")
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.view_init(elev=20, azim=-160)
        if positions:
            pts = np.array(positions)
            g_min, g_max = pts.min(), pts.max()
            center = (g_min + g_max) / 2
            r = g_max - g_min
            pad = r * 0.1 + 0.05
            for setter in [ax.set_xlim, ax.set_ylim, ax.set_zlim]:
                setter(center - r / 2 - pad, center + r / 2 + pad)
        lines = [ax.plot([], [], [], c, linewidth=2, alpha=0.8)[0] for c in colors]
        lbls = [
            ax.text(0, 0, 0, lbl, color=c, fontsize=8, weight="bold")
            for lbl, c in zip(labels, colors)
        ]
        return ax, lines, lbls

    _, left_axis_lines, left_axis_labels = _setup_3d_ax(
        gs[1, 1],
        "Left Gripper 3D Position & Orientation",
        all_left_positions,
        ["r", "g", "b"],
        ["L_X", "L_Y", "L_Z"],
    )
    ax_3d_right, right_axis_lines, right_axis_labels = _setup_3d_ax(
        gs[1, 2],
        "Right Gripper 3D Position & Orientation",
        all_right_positions,
        ["m", "c", "y"],
        ["R_X", "R_Y", "R_Z"],
    )

    ax_meta = fig.add_subplot(gs[2, :])
    ax_meta.text(
        0.1,
        0.5,
        f"UID: {frame_data['uids']}\nDataset: {frame_data['dataset_names']}\n"
        f"Frame: {frame_data['frame_index']}\nAction Shape: {action.shape}\n"
        f"Videos: {len(videos)} cameras",
        transform=ax_meta.transAxes,
        fontsize=10,
        verticalalignment="center",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8),
    )
    ax_meta.set_title("Metadata", fontsize=12)
    ax_meta.axis("off")

    ax_text = fig.add_subplot(gs[3, :])
    ax_text.text(
        0.05,
        0.5,
        f"Text Sequence:\n{instruction_text}",
        transform=ax_text.transAxes,
        fontsize=10,
        verticalalignment="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8),
    )
    ax_text.set_title("Text Sequence", fontsize=12)
    ax_text.axis("off")

    fig.suptitle(
        "Comprehensive Video Animation - Robot Action Prediction",
        fontsize=16,
        fontweight="bold",
        y=0.95,
    )
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.9)

    def animate(frame_idx):
        all_artists = []
        for i, video_np in enumerate(processed_videos):
            fi = min(frame_idx, video_np.shape[0] - 1)
            frame_np = video_np[fi]
            if len(frame_np.shape) == 3 and frame_np.shape[0] == 3:
                frame_np = np.transpose(frame_np, (1, 2, 0))
            if frame_np.max() > 1.0:
                frame_np = frame_np / 255.0
            video_ims[i].set_array(frame_np)
            all_artists.append(video_ims[i])

        if frame_idx < action.shape[0]:
            frame_line_1.set_xdata([frame_idx * interval, frame_idx * interval])
            frame_line_2.set_xdata([frame_idx * interval, frame_idx * interval])
            all_artists.extend([frame_line_1, frame_line_2])

            t = min(frame_idx * interval, action.shape[0] - 1)
            if action.shape[1] >= 14:
                for pos_slice, euler_slice, axis_lines, axis_labels in [
                    (action[t, 0:3], action[t, 3:6], left_axis_lines, left_axis_labels),
                    (
                        action[t, 7:10],
                        action[t, 10:13],
                        right_axis_lines,
                        right_axis_labels,
                    ),
                ]:
                    try:
                        rot = Rotation.from_euler(
                            "xyz", euler_slice, degrees=False
                        ).as_matrix()
                        basis = rot @ (np.eye(3) * axis_len)
                        for line, vec, lbl in zip(axis_lines, basis.T, axis_labels):
                            line.set_data(
                                [pos_slice[0], pos_slice[0] + vec[0]],
                                [pos_slice[1], pos_slice[1] + vec[1]],
                            )
                            line.set_3d_properties(
                                [pos_slice[2], pos_slice[2] + vec[2]]
                            )
                            lbl.set_position(
                                (pos_slice[0] + vec[0], pos_slice[1] + vec[1])
                            )
                            lbl.set_3d_properties(pos_slice[2] + vec[2], zdir="z")
                        all_artists.extend(axis_lines)
                        all_artists.extend(axis_labels)
                    except (ValueError, RuntimeError):
                        pass
        return tuple(all_artists)

    anim = FuncAnimation(
        fig, animate, frames=max_frames, interval=1000 / fps, blit=False, repeat=True
    )
    print(f"Creating comprehensive animated GIF with {max_frames} frames...")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    anim.save(save_path, writer="pillow", fps=fps, dpi=120)
    print(f"Comprehensive animated GIF saved to: {save_path}")
    plt.close(fig)
    return anim
