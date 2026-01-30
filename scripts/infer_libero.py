import argparse
import time
import os
# from wall_x.utils.baseline_utils import check_baseline_dump, update_baseline
from wall_x.infer.utils_libero import set_seed_everywhere, TaskSuite, TASK_MAX_STEPS
from wall_x.infer.infer_config import InferConfig
from wall_x.infer.env_libero import LiberoRobotEnv


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Wall-X Libero evaluation script")
    args.add_argument("--seed", type=int, default=42, help="Random seed")
    args.add_argument("--id", type=int, default=None, help="Unique index id")
    args.add_argument("--name", type=int, default=None, help="Launch command name")
    args.add_argument(
        "--baseline_path", type=str, default=None, help="Path to baseline record table"
    )
    args.add_argument(
        "--update_baseline", type=bool, default=False, help="Whether to update baseline table"
    )
    args.add_argument(
        "--mode", type=str, default="flow", choices=["flow", "ar"], help="Running mode"
    )
    args.add_argument(
        "--checkpoint_path", type=str, required=True, help="Model checkpoint path"
    )
    args.add_argument(
        "--train_config_path",
        type=str,
        required=False,
        default=None,
        help="Path to training config .yml file",
    )
    args.add_argument(
        "--norm_key",
        type=str,
        default="physical-intelligence/libero",
        help="Key for normalization statistics",
    )
    args.add_argument(
        "--cam_names",
        nargs="+",
        default=["face_view", "right_wrist_view"],
        help="List of camera names (e.g., --cam_names face_view right_wrist_view)",
    )
    args.add_argument(
        "--task_suite_name",
        type=str,
        default=TaskSuite.LIBERO_SPATIAL,
        choices=[e.value for e in TaskSuite],
        help="Libero task suite to load",
    )
    args.add_argument(
        "--initial_states_path",
        type=str,
        default="DEFAULT",
        help="Path to initial states .json file, or 'DEFAULT' to use default states.",
    )
    args.add_argument(
        "--num_trials_per_task",
        type=int,
        default=50,
        help="Number of evaluation episodes to run per task",
    )
    args.add_argument(
        "--rollout_dir", type=str, default="./rollouts", help="Directory to save rollout videos"
    )
    args = args.parse_args()

    print(f"Using random seed: {args.seed}")
    set_seed_everywhere(args.seed)

    print("Initializing InferConfig...")
    if args.train_config_path is None:
        args.train_config_path = os.path.join(args.checkpoint_path, "config.yml")

    config = InferConfig(
        checkpoint_path=args.checkpoint_path,
        train_config_path=args.train_config_path,
        norm_key=args.norm_key,
        cam_names=args.cam_names,
    )
    if args.mode == "flow":
        config.action_horizon = config.train_config.get("data", {}).get(
            "action_horizon_flow", 10
        )
    elif args.mode == "ar":
        config.action_horizon = config.train_config.get("data", {}).get(
            "action_horizon_ar", 10
        )
    else:
        raise ValueError(f"Invalid mode: {args.mode}")
    config.model_device = "cuda"

    print("Initializing LiberoRobotEnv (Evaluator)...")

    config.action_dim=7
    config.pred_horizon=10

    evaluator = LiberoRobotEnv(
        config=config,
        task_suite_name=args.task_suite_name,
        initial_states_path=args.initial_states_path,
        rollout_dir=args.rollout_dir,
        seed=args.seed,
    )

    print(f"\n{'='*20} Starting Evaluation {'='*20}")
    print(f"Task suite: {args.task_suite_name}")
    print(f"Number of tasks: {evaluator.num_tasks}")
    print(f"Trials per task: {args.num_trials_per_task}")
    print(f"Initial states: {args.initial_states_path}")
    print(f"Videos will be saved to: {evaluator.rollout_dir}")
    print(f"{'='*50}\n")

    total_successes = 0
    total_episodes_run = 0
    start_time = time.time()

    for task_id in range(evaluator.num_tasks):
        task_successes = 0
        task_episodes_attempted = 0

        libero_env_instance = None
        task_desc = ""
        initial_states = None

        max_infer_times = TASK_MAX_STEPS[args.task_suite_name]
        print(
            f"{args.task_suite_name} TASK_MAX_STEPS: {TASK_MAX_STEPS[args.task_suite_name]}"
        )
        for ep_idx in range(args.num_trials_per_task):
            print(f"  > Running trial {ep_idx + 1} / {args.num_trials_per_task}...")

            try:
                print(f"\nCreating environment for Task {task_id}...")
                libero_env_instance, task_desc, initial_states = (
                    evaluator.create_env_for_task(task_id)
                )
                print(
                    f"--- Starting task {task_id + 1} / {evaluator.num_tasks}: {task_desc} ---"
                )
            except Exception as e:
                print(
                    f"\n[CRITICAL ERROR] Failed to create environment for task {task_id}: {e}. Skipping entire task."
                )
                continue

            task_episodes_attempted += 1
            total_episodes_run += 1

            success = False
            try:
                if args.mode == "flow":
                    success = evaluator.run_infer_flow_action(
                        env=libero_env_instance,
                        task_id=task_id,
                        task_desc=task_desc,
                        default_initial_states=initial_states,
                        episode_idx=ep_idx,
                        max_infer_times=max_infer_times,
                    )
                elif args.mode == "ar":
                    success = evaluator.run_infer_ar_action(
                        env=libero_env_instance,
                        task_id=task_id,
                        task_desc=task_desc,
                        default_initial_states=initial_states,
                        episode_idx=ep_idx,
                        max_infer_times=max_infer_times,
                    )
            except Exception as e:
                print(f"  [EXCEPTION] Episode run error: {e}")

            if success:
                task_successes += 1
                total_successes += 1
                print("  > Trial result: SUCCESS")
            else:
                print("  > Trial result: FAILURE")

            if task_episodes_attempted > 0:
                print(
                    f"  > Task {task_id} current success rate: {task_successes / task_episodes_attempted * 100:.1f}% ({task_successes}/{task_episodes_attempted})"
                )
            if total_episodes_run > 0:
                print(
                    f"  > Overall current success rate: {total_successes / total_episodes_run * 100:.1f}% ({total_successes}/{total_episodes_run})"
                )

        task_success_rate = (
            task_successes / task_episodes_attempted
            if task_episodes_attempted > 0
            else 0
        )
        print(f"\n--- Task {task_id} ({task_desc}) Summary ---")
        print(
            f"Success rate: {task_success_rate * 100:.1f}% ({task_successes}/{task_episodes_attempted})"
        )
        print(f"{'-'*40}\n")

    end_time = time.time()
    total_time = end_time - start_time
    final_success_rate = (
        total_successes / total_episodes_run if total_episodes_run > 0 else 0
    )

    print(f"\n{'='*20} Final Evaluation Summary {'='*20}")
    print(f"Total runtime: {total_time:.2f} seconds ({total_time / 60:.1f} minutes)")
    print(f"Total trials run: {total_episodes_run}")
    print(f"Total successes: {total_successes}")
    print(f"Overall success rate: {final_success_rate * 100:.2f}%")
    print(f"{'='*56}")

    print("Evaluation completed.")

