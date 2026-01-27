#!/bin/bash
export PYTHONPATH="/mnt/data/x2robot_v2/yangping/gitlab/LIBERO:${PYTHONPATH}"
# ================= 配置区域 =================
# checkpoint_path="/x2robot_v2/share/ryan/ckpts/libero_bus_flow_moe/24_50000"
checkpoint_path="/x2robot_v2/share/yangping/ckpt/robochallenge_github/benchmark/libero/bus2507_plus_moe-0126/20"
mode="flow"
norm_key="libero_all"

# checkpoint_path="/x2robot_v2/share/xzn/libero/3"
# mode="ar"
# norm_key="libero_all"

initial_states_path="DEFAULT"
num_trials_per_task=10
seed=3407
rollout_dir="./rollouts/eval_${mode}_${seed}_$(date +%Y%m%d_%H%M%S)"
cam_names=("face_view" "right_wrist_view")

# 日志保存目录
log_dir="./logs/eval_${mode}_${seed}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$log_dir"

# ================= 并行任务定义 =================

# 定义 4 个子任务套件名称
task_suites=("libero_spatial" "libero_object" "libero_goal" "libero_10")

# 定义对应的 4 个 GPU ID
gpu_ids=(0 1 2 3)

# ================= 开始执行 =================

echo "开始并行评测..."
echo "日志将保存至: $log_dir"

# 存放 PID 的数组
pids=()

len=${#task_suites[@]}
for (( i=0; i<$len; i++ )); do
    suite_name=${task_suites[$i]}
    cuda_id=${gpu_ids[$i]}
    log_file="$log_dir/${suite_name}_gpu${cuda_id}.log"

    echo "正在启动任务: $suite_name on GPU $cuda_id (Log: $log_file)"

    CUDA_VISIBLE_DEVICES=$cuda_id python scripts/infer_libero.py \
        --seed "$seed" \
        --mode "$mode" \
        --checkpoint_path "$checkpoint_path" \
        --norm_key "$norm_key" \
        --cam_names "${cam_names[@]}" \
        --task_suite_name "$suite_name" \
        --initial_states_path "$initial_states_path" \
        --num_trials_per_task "$num_trials_per_task" \
        --rollout_dir "$rollout_dir" \
        > "$log_file" 2>&1 &

    # 记录 PID
    pids+=($!)
done

# ================= 等待结束 =================

echo "所有任务已在后台启动。"
echo "正在等待所有任务完成..."
echo "可以使用 'tail -f $log_dir/*.log' 查看实时进度。"

# 打印 kill 命令提示
echo "如需终止所有进程，请执行："
echo "kill -9 ${pids[*]}"

wait

echo "所有评测任务已完成！"
