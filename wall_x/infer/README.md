# Wall-X 推理系统使用指南

本文档介绍如何使用 Wall-X 机器人推理系统进行模型部署和实时推理。

## 📋 目录

- [系统架构](#系统架构)
- [快速开始](#快速开始)
- [核心组件](#核心组件)
- [配置说明](#配置说明)
- [使用示例](#使用示例)
- [仿真推理](#仿真推理)
  - [LIBERO BENCHMARK](#libero-benchmark)
    - [基础推理](#基础推理)
    - [Batch Inference（批量推理）](#batch-inference批量推理)
    - [Ray 数据并行推理](#ray-数据并行推理)
- [常见问题](#常见问题)

---

## 🏗️ 系统架构

推理系统由以下核心组件组成：

```
wall_x/infer/
├── infer_config.py      # 推理配置管理
├── env.py               # 环境封装（BaseEnv, RealRobotEnv）
├── robot.py             # 机器人控制（DesktopRobot, TurtleRobot）
├── model_wrapper.py     # 模型封装和推理
├── base_dataclass.py    # 数据结构定义
├── socket_controller.py # Socket通信控制
├── logger.py            # 日志系统
└── utils.py             # 工具函数
```

---

## 🚀 快速开始

### 1. 基础使用

请参考[infer new](run_scripts/infer_new.py)，最简单的使用方式是直接调用已经实现好的推理流程。你只需要配置对[infer config](wall_x/infer/infer_config.py)的相关参数就可以直接使用：

```python
from wall_x.infer.infer_config import InferConfig
from wall_x.infer.env import RealRobotEnv

# 创建配置
config = InferConfig()
config.robot_port = 32006
config.model_device = "cuda:2"

# 定义任务指令
instructions = [
    "Pick up the cup.",
    "Pick up the apple.",
    "Pick up the banana.",
]

# 创建环境并运行
env = RealRobotEnv(config, instructions)
env.run_infer_flow_action()  # 直接运行flow action推理
```

### 2. 自定义推理流程

你也可以自定义推理流程以满足特定需求：

```python
def run_infer_flow_action_with_subtask(env: RealRobotEnv):
    while True:
        # 监听键盘控制
        if env.listen_to_keyboard():
            continue

        # 获取观察和指令
        observation = env.get_observation()
        instruction = env.get_instruction()

        # 先生成子任务
        subtask = env.model.infer_subtask(observation, instruction)

        # 基于子任务生成动作
        model_output = env.model.infer_flow_action(observation, subtask)

        # 执行动作
        env.apply_action(model_output)

# 使用自定义流程
env = RealRobotEnv(config, instructions)
run_infer_flow_action_with_subtask(env)
```

---

## 🔧 核心组件

### 1. InferConfig - 推理配置

`InferConfig` 管理所有推理相关的配置参数。

**主要配置项：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `checkpoint_path` | str | 必填 | 模型checkpoint路径 |
| `train_config_path` | str | None | 训练配置文件路径（为None时自动从checkpoint读取） |
| `robot_host` | str | '0.0.0.0' | 机器人通信地址 |
| `robot_port` | int | 33723 | 机器人通信端口 |
| `robot_type` | str | 'desktop' | 机器人类型（desktop/turtle） |
| `robot_action_start_ratio` | float | 0 | 动作序列起始比例 |
| `robot_action_end_ratio` | float | 0.8 | 动作序列结束比例 |
| `robot_action_interpolate_multiplier` | int | 70 | 动作插值倍数 |
| `robot_use_joint_angle_control` | bool | False | 是否使用关节角度控制 |
| `turtle_as_desktop` | bool | False | 使用乌龟本体做桌面操作，固定底盘头移动，头部相机，和底盘高度 |
| `action_horizon` | int | 32 | 动作序列长度 |
| `model_device` | str | 'cuda:2' | 模型运行设备 |
| `num_inference_timesteps` | int | 10 | Flow matching推理步数 |

**示例：**

```python
config = InferConfig()
config.checkpoint_path = '/path/to/your/checkpoint'
config.robot_port = 32006
config.robot_type = 'desktop'
config.model_device = 'cuda:0'
config.robot_action_end_ratio = 0.7
```

### 2. RealRobotEnv - 机器人环境

`RealRobotEnv` 封装了机器人控制和模型推理的完整环境。

**主要方法：**

- `get_observation()` - 获取当前观察（相机图像 + 机器人状态）
- `get_instruction()` - 获取当前任务指令
- `apply_action(model_output)` - 执行模型输出的动作
- `reset()` - 重置机器人到初始状态
- `listen_to_keyboard()` - 监听键盘控制指令

**内置推理方法：**

- `run_infer_flow_action()` - 运行flow action推理循环

### 3. WallxModelWrapper - 模型封装

`WallxModelWrapper` 封装了模型加载和推理功能。

**这里我们已经实现了三个主要的方法，满足一般的需求：**

| 方法 | 参数 | 返回值 | 说明 |
|------|------|--------|------|
| `infer_flow_action()` | observation, instruction | model_output | Flow matching方式生成动作 |
| `infer_ar_action()` | observation, instruction | model_output | 自回归方式生成动作 |
| `infer_subtask()` | observation, instruction | subtask (str) | 生成自然语言子任务 |

**示例：**

```python
# 通过env获取模型
observation = env.get_observation()
instruction = "Pick up the red cup"

# Flow action推理
flow_output = env.model.infer_flow_action(observation, instruction)
env.apply_action(flow_output)

# 子任务生成
subtask = env.model.infer_subtask(observation, instruction)
print(f"Generated subtask: {subtask}")

# AR action推理
ar_output = env.model.infer_ar_action(observation, instruction)
env.apply_action(ar_output)
```

### 4. Robot类 - 机器人控制

系统支持两种机器人类型：

**DesktopRobot** - 桌面机器人
- 双臂控制（笛卡尔坐标或关节角度）
- 手爪控制

**TurtleRobot** - 移动机器人
- 除了双臂和手爪外，还支持：
  - 头部控制
  - 升降控制
  - 底盘运动控制

**Debug模式 - 不连接真实机器人**

如果需要在没有真实机器人的情况下进行代码调试和测试，可以使用 `DummyRobotController`。在 `robot.py` 中（第34-45行）切换控制器：

```python
# 测试/Debug模式
self.robot_controller = DummyRobotController(
    robot_id=robot_id,
    host=config.robot_host,
    port=config.robot_port
)

# 实际推理模式（取消下面的注释，注释掉上面的代码）
# self.robot_controller = RobotController(
#     robot_id=robot_id,
#     host=config.robot_host,
#     port=config.robot_port
# )
```

使用 `DummyRobotController` 时，系统会模拟机器人的响应，允许你测试推理流程而无需连接真实硬件。

---

## ⚙️ 配置说明

### 机器人类型选择

```python
# Desktop机器人
config.robot_type = "desktop"

# Turtle机器人
config.robot_type = "turtle"
```

### 动作控制模式

```python
# 笛卡尔坐标控制（默认）
config.robot_use_joint_angle_control = False

# 关节角度控制
config.robot_use_joint_angle_control = True
```

### 动作执行参数调优

动作执行有三个关键参数：

```python
# 1. 动作序列截取范围
config.robot_action_start_ratio = 0.0   # 从0%开始
config.robot_action_end_ratio = 0.8     # 到80%结束

# 2. 插值倍数（越大越平滑，但执行越慢）
config.robot_action_interpolate_multiplier = 70
```

---

## 💡 使用示例

### 示例1：基础推理

```python
from wall_x.infer.infer_config import InferConfig
from wall_x.infer.env import RealRobotEnv

config = InferConfig()
config.checkpoint_path = '/path/to/checkpoint'
config.robot_port = 32006
config.model_device = "cuda:0"

instructions = ["Pick up the object."]
env = RealRobotEnv(config, instructions)
env.run_infer_flow_action()
```

### 示例2：多任务切换

```python
instructions = [
    "Pick up the cup.",          # 任务0
    "Place it on the table.",    # 任务1
    "Pick up the apple.",        # 任务2
]

env = RealRobotEnv(config, instructions)

# 程序运行时，按键盘数字键0-2可切换任务
env.run_infer_flow_action()
```

### 示例3：子任务分解 + Flow Action

```python
def run_with_subtask(env: RealRobotEnv):
    while True:
        if env.listen_to_keyboard():
            continue

        observation = env.get_observation()
        instruction = env.get_instruction()

        # 先分解子任务
        subtask = env.model.infer_subtask(observation, instruction)
        print(f"Subtask: {subtask}")

        # 基于子任务执行
        model_output = env.model.infer_flow_action(observation, subtask)
        env.apply_action(model_output)

env = RealRobotEnv(config, instructions)
run_with_subtask(env)
```

### 示例4：Turtle机器人全功能控制

```python
config = InferConfig()
config.robot_type = "turtle"  # 使用turtle机器人
config.robot_port = 32010

instructions = [
    "Navigate to the kitchen and pick up the cup.",
    "Move to the table and place the cup.",
]

env = RealRobotEnv(config, instructions)
env.run_infer_flow_action()  # 自动支持底盘移动 + 双臂控制
```

---

## 🎮 键盘控制

推理过程中支持键盘实时控制：

| 按键 | 功能 |
|------|------|
| `r` | 重置机器人到初始位置 |
| `s` | 停止推理并直到再次按下s |
| `0-9` | 切换到对应索引的指令 |


如需禁用键盘控制：
```python
env = RealRobotEnv(config, instructions, enable_keyboard=False)
```

---

## 仿真推理

### LIBERO BENCHMARK

环境安装

```bash
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
pip install -e LIBERO
pip install "imageio[ffmpeg]" robosuite==1.4.1 bddl easydict cloudpickle gym
```

#### 基础推理

执行推理（执行bash run_libero.sh之前要先运行下面这个，把默认的libero路径设置好）

```bash
checkpoint_path="/x2robot_v2/share/xzn/libero/3"
mode="ar" # 选择模式 'flow' or 'ar'
norm_key="libero_all" # 指定normalizer的key

cuda_id=3
task_suite_name="libero_spatial" # 四个子集：libero_spatial, libero_object, libero_goal, libero_10
initial_states_path="DEFAULT"
num_trials_per_task=50 # 每个任务的重复次数
rollout_dir="./rollouts" # 输出回放视频路径

export CUDA_VISIBLE_DEVICES=$cuda_id
python run_scripts/infer_libero.py \
    --checkpoint_path $checkpoint_path \
    --task_suite_name $task_suite_name \
    --initial_states_path $initial_states_path \
    --num_trials_per_task $num_trials_per_task \
    --rollout_dir $rollout_dir \
    --norm_key $norm_key \
    --mode $mode
```

#### Batch Inference（批量推理）

使用 `infer_libero_batch.py` 脚本可以启用批量推理功能，通过 `--batch_size` 参数控制模型推理的批次大小。**注意**：批量推理仅在 `flow` 模式下有效，环境步进仍为串行执行。

**优势**：
- 提高 GPU 利用率，加速模型推理
- 适合需要运行大量 episode 的场景

**示例**：

```bash
checkpoint_path="/x2robot_v2/share/ryan/ckpts/libero_bus_flow_moe/24_50000"
mode="flow" # batch inference 仅支持 flow 模式
norm_key="libero_all"

cuda_id=3
task_suite_name="libero_spatial"
initial_states_path="DEFAULT"
num_trials_per_task=50
rollout_dir="./rollouts_batch"
batch_size=16 # 批量推理的样本数量

export CUDA_VISIBLE_DEVICES=$cuda_id
export ENABLE_EXPERIMENTAL_INFERENCE_ENGINE=True
export ENABLE_CUDA_GRAPH=True
export USE_FLASHINFER=True

python run_scripts/infer_libero_batch.py \
    --checkpoint_path $checkpoint_path \
    --task_suite_name $task_suite_name \
    --initial_states_path $initial_states_path \
    --num_trials_per_task $num_trials_per_task \
    --rollout_dir $rollout_dir \
    --norm_key $norm_key \
    --mode $mode \
    --batch_size $batch_size
```

**参数说明**：
- `--batch_size`: 模型推理的批次大小（默认：1）。增大此值可以提高 GPU 利用率，但需要更多显存。

#### Ray 数据并行推理

使用 `--use_ray` 参数可以启用 Ray 分布式数据并行推理。每个 worker 会独立加载模型并在不同的 GPU 上并行运行 episode，大幅提升推理吞吐量。

**优势**：
- 多 GPU 并行，充分利用硬件资源
- 每个 worker 独立运行，避免资源竞争
- 适合大规模评测任务

**前置要求**：
```bash
pip install ray
```

**示例**：

```bash
checkpoint_path="/x2robot_v2/share/ryan/ckpts/libero_bus_flow_moe/24_50000"
mode="flow" # Ray 并行目前仅支持 flow 模式
norm_key="libero_all"

cuda_id="0,1,2,3" # 指定多个 GPU
task_suite_name="libero_spatial"
initial_states_path="DEFAULT"
num_trials_per_task=50
rollout_dir="./rollouts_ray"

# Ray 配置
num_workers=4                    # Ray worker 数量
ray_num_gpus_per_worker=1.0      # 每个 worker 申请的 GPU 数量
ray_chunk_size=32                # 每个 remote 调用并行推理的 episode 数量，相当于单实例上推理的batch_size
ray_rollout_dir_mode="per_worker" # rollout 保存目录策略：per_worker 或 shared

export CUDA_VISIBLE_DEVICES=$cuda_id
export ENABLE_EXPERIMENTAL_INFERENCE_ENGINE=True
export ENABLE_CUDA_GRAPH=True
export USE_FLASHINFER=True

python run_scripts/infer_libero_batch.py \
    --checkpoint_path $checkpoint_path \
    --task_suite_name $task_suite_name \
    --initial_states_path $initial_states_path \
    --num_trials_per_task $num_trials_per_task \
    --rollout_dir $rollout_dir \
    --norm_key $norm_key \
    --mode $mode \
    --use_ray \
    --num_workers $num_workers \
    --ray_num_gpus_per_worker $ray_num_gpus_per_worker \
    --ray_chunk_size $ray_chunk_size \
    --ray_rollout_dir_mode $ray_rollout_dir_mode \
    --disable_video  # 可选：禁用视频保存以提升速度
```

**Ray 参数说明**：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--use_ray` | flag | False | 启用 Ray 数据并行推理 |
| `--num_workers` | int | 4 | Ray worker 数量，建议等于可用 GPU 数量 |
| `--ray_num_gpus_per_worker` | float | 1.0 | 每个 worker 申请的 GPU 数量（支持小数） |
| `--ray_chunk_size` | int | 1 | 每个 remote 调用并行推理的 episode 数量，增大提高 GPU 利用率 |
| `--ray_rollout_dir_mode` | str | "per_worker" | 保存目录策略：`per_worker`（每 worker 单独子目录）或 `shared`（共享目录） |
| `--ray_address` | str | None | Ray 集群地址（可选，用于连接远程集群） |
| `--disable_video` | flag | False | 禁用 rollout 视频保存，可显著提升速度 |

**注意事项**：
- Ray 并行目前仅支持 `flow` 模式
- 确保 `num_workers * ray_num_gpus_per_worker <= 可用 GPU 数量`
- 单卡多 worker 可能导致 OOM 或性能下降，不推荐
- 使用 `per_worker` 模式可以避免多 worker 写入同一目录的冲突
- 大规模评测时建议使用 `--disable_video` 提升速度

---


## 🐛 常见问题

### Q1: 如果我有一个想实现的功能如何改动代码：
1.首先请确定model层面是否支持，主要看能否基于现有的infer函数实现。如果不能请先自定义infer函数，这里需要注意，模型预测的数值需要被添加到统一的动作管理类中：
```python
model_output["robot_state_action_data"] = observation["robot_state_action_data"]
model_output["robot_state_action_data"].save_action_data(model_output['predict_action'])
```
只有成功添加，你的动作才能被机器人测的数据处理感知到。

2.在确认model层面可以支持后，请编写对应的run函数，理论上应该在run scirpts中实现。如果功能测试稳定后可以提PR加入默认的run函数中。
