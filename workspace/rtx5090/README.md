# RTX 5090 推理部署

本目录包含在 **NVIDIA RTX 5090** 上部署 Wall-X 推理服务的脚本与文档。

## 目录结构

```
workspace/rtx5090/
├── README.md          # 本文件
├── DEPLOY.md          # 通用部署指南（任意 5090 机器）
├── install.sh         # 环境安装脚本
├── run_server.sh      # 启动推理服务
├── env.example        # 路径配置模板
└── local/             # 本机专用配置（可按需修改，勿提交敏感路径）
    ├── env.sh         # 本机路径
    └── NOTES.md       # 本机实测记录
```

## 快速开始

**1. 配置路径**

```bash
cp workspace/rtx5090/env.example workspace/rtx5090/local/env.sh
# 编辑 local/env.sh，填入 CONDA_SH、CONDA_ENV、LEROBOT_DIR、CHECKPOINT_PATH
```

**2. 安装环境**（在仓库根目录执行）

```bash
bash workspace/rtx5090/install.sh
```

**3. 启动服务**

```bash
bash workspace/rtx5090/run_server.sh
```

**4. 健康检查**

```bash
curl http://127.0.0.1:44660/healthz   # 期望: OK
```

## 文档说明

| 文件 | 用途 |
|------|------|
| [DEPLOY.md](./DEPLOY.md) | 通用部署流程，使用占位符路径，适合任意 5090 推理机 |
| [local/NOTES.md](./local/NOTES.md) | 本机实测记录（环境信息、踩坑、验证结果） |

## 环境变量

脚本通过 `local/env.sh` 加载以下变量：

| 变量 | 说明 |
|------|------|
| `CONDA_SH` | conda 初始化脚本路径 |
| `CONDA_ENV` | conda 环境名或 `--prefix` 全路径 |
| `LEROBOT_DIR` | LeRobot 源码目录 |
| `CHECKPOINT_PATH` | 模型 checkpoint 目录 |
| `PORT` | 服务端口（默认 44660） |
| `CUDA_ID` | GPU 编号（默认 0） |

服务启动时还会设置 `ENABLE_FAST_PREPROCESS=false`、`ENABLE_CUDA_GRAPH=1` 等，详见 [DEPLOY.md](./DEPLOY.md)。
