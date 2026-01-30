import argparse
import time
import os
# from wall_x.utils.baseline_utils import check_baseline_dump, update_baseline
from wall_x.infer.utils_libero import set_seed_everywhere, TaskSuite, TASK_MAX_STEPS
from wall_x.infer.infer_config import InferConfig
from wall_x.infer.env_libero import LiberoRobotEnv


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Wall-X Libero evaluation script")
    args.add_argument("--seed", type=int, default=42, help="随机种子")
    args.add_argument("--id", type=int, default=None, help="唯一索引id")
    args.add_argument("--name", type=int, default=None, help="启动命令名称")
    args.add_argument(
        "--baseline_path", type=str, default=None, help="baseline记录表格的路径"
    )
    args.add_argument(
        "--update_baseline", type=bool, default=False, help="是否更新baseline表格"
    )
    args.add_argument(
        "--mode", type=str, default="flow", choices=["flow", "ar"], help="运行模式"
    )
    args.add_argument(
        "--checkpoint_path", type=str, required=True, help="模型检查点路径"
    )
    args.add_argument(
        "--train_config_path",
        type=str,
        required=False,
        default=None,
        help="训练配置 .yml 文件路径",
    )
    args.add_argument(
        "--norm_key",
        type=str,
        default="physical-intelligence/libero",
        help="归一化统计的键",
    )
    args.add_argument(
        "--cam_names",
        nargs="+",
        default=["face_view", "right_wrist_view"],
        help="采用的相机名称列表 (例如: --cam_names face_view right_wrist_view)",
    )
    args.add_argument(
        "--task_suite_name",
        type=str,
        default=TaskSuite.LIBERO_SPATIAL,
        choices=[e.value for e in TaskSuite],
        help="要加载的 Libero 任务套件",
    )
    args.add_argument(
        "--initial_states_path",
        type=str,
        default="DEFAULT",
        help="初始状态 .json 文件的路径，或 'DEFAULT' 使用默认状态。",
    )
    args.add_argument(
        "--num_trials_per_task",
        type=int,
        default=50,
        help="每个任务要运行的评测 episode 数量",
    )
    args.add_argument(
        "--rollout_dir", type=str, default="./rollouts", help="保存 rollout 视频的目录"
    )
    args = args.parse_args()

    print(f"使用随机种子: {args.seed}")
    set_seed_everywhere(args.seed)

    print("正在初始化 InferConfig...")
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

    print("正在初始化 LiberoRobotEnv (Evaluator)...")

    config.action_dim=7
    config.pred_horizon=10

    evaluator = LiberoRobotEnv(
        config=config,
        task_suite_name=args.task_suite_name,
        initial_states_path=args.initial_states_path,
        rollout_dir=args.rollout_dir,
        seed=args.seed,
    )

    print(f"\n{'='*20} 开始评测 {'='*20}")
    print(f"任务套件: {args.task_suite_name}")
    print(f"任务数量: {evaluator.num_tasks}")
    print(f"每个任务的试验次数: {args.num_trials_per_task}")
    print(f"初始状态: {args.initial_states_path}")
    print(f"视频将保存至: {evaluator.rollout_dir}")
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
            print(f"  > 运行试验 {ep_idx + 1} / {args.num_trials_per_task}...")

            try:
                print(f"\n正在为 Task {task_id} 创建环境...")
                libero_env_instance, task_desc, initial_states = (
                    evaluator.create_env_for_task(task_id)
                )
                print(
                    f"--- 开始任务 {task_id + 1} / {evaluator.num_tasks}: {task_desc} ---"
                )
            except Exception as e:
                print(
                    f"\n[严重错误] 无法创建任务 {task_id} 的环境: {e}。跳过整个任务。"
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
                print(f"  [异常] Episode 运行出错: {e}")

            if success:
                task_successes += 1
                total_successes += 1
                print("  > 试验结果: 成功 (SUCCESS)")
            else:
                print("  > 试验结果: 失败 (FAILURE)")

            if task_episodes_attempted > 0:
                print(
                    f"  > 任务 {task_id} 即时成功率: {task_successes / task_episodes_attempted * 100:.1f}% ({task_successes}/{task_episodes_attempted})"
                )
            if total_episodes_run > 0:
                print(
                    f"  > 总体即时成功率: {total_successes / total_episodes_run * 100:.1f}% ({total_successes}/{total_episodes_run})"
                )

        task_success_rate = (
            task_successes / task_episodes_attempted
            if task_episodes_attempted > 0
            else 0
        )
        print(f"\n--- 任务 {task_id} ({task_desc}) 总结 ---")
        print(
            f"成功率: {task_success_rate * 100:.1f}% ({task_successes}/{task_episodes_attempted})"
        )
        print(f"{'-'*40}\n")

    end_time = time.time()
    total_time = end_time - start_time
    final_success_rate = (
        total_successes / total_episodes_run if total_episodes_run > 0 else 0
    )

    print(f"\n{'='*20} 最终评测总结 {'='*20}")
    print(f"总运行时间: {total_time:.2f} 秒 ({total_time / 60:.1f} 分钟)")
    print(f"总运行试验次数: {total_episodes_run}")
    print(f"总成功次数: {total_successes}")
    print(f"整体成功率: {final_success_rate * 100:.2f}%")
    print(f"{'='*56}")

    print("评测完成。")

