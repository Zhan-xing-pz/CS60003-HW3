# Topic 2: LeRobot ACT 跨环境泛化

本目录实现题目二“基于 LeRobot 的 ACT 策略跨环境泛化挑战”。

实验比较：

- `act_calvin_A`：仅使用 CALVIN 环境 A 训练。
- `act_calvin_ABC`：联合使用环境 A/B/C 训练。

两个模型使用相同的 ACT 架构和训练超参数，只根据训练域 validation Action L1 选择最佳 checkpoint；环境 D 仅用于最终 zero-shot 评估。

## 最终设置

| 项目 | 设置 |
|---|---|
| Training Steps | 30000 |
| Batch Size | 64 |
| Learning Rate | 1e-4 |
| Optimizer | AdamW |
| Weight Decay | 1e-6 |
| Loss | L1 imitation loss |
| ACT Chunk Size | 16 |
| Checkpoint Frequency | 1000 steps |
| Validation per Checkpoint | 50 batches |
| Final D Evaluation | 200 batches |

最终结果：

| 模型 | Best Step | Validation Action L1 | D Zero-shot Action L1 |
|---|---:|---:|---:|
| A-only | 14000 | 0.475544 | 0.496255 |
| A/B/C joint | 27000 | 0.427793 | 0.447098 |

D 的离线数据没有可靠任务完成信号，因此使用 Action L1 作为主要 zero-shot 指标。

## 目录结构

```text
configs/                ACT 训练与 smoke test 配置
scripts/                数据准备、训练、评估与绘图脚本
src/hw3_act/            核心实现
requirements.txt        Python 依赖
environment.yml         Conda 环境
outputs/                本地运行结果，不提交 Git
wandb/                  本地 WandB 记录，不提交 Git
model_weights/          本地整理后的最佳模型副本，不提交 Git
```

## 环境

正式实验使用服务器 Conda 环境：

```bash
conda activate pz
```

新环境可执行：

```bash
conda env create -f environment.yml
conda activate hw3-act
```

环境文件已锁定实验实际使用的 `lerobot==0.4.4`。也可以在已有 Python
3.10 环境中执行：

```bash
pip install -r requirements.txt
```

脚本默认使用当前激活环境的 Python，并从 `PATH` 查找
`lerobot-train`。特殊部署可通过 `HW3_PYTHON` 和
`HW3_LEROBOT_TRAIN` 覆盖可执行文件路径，不依赖原实验服务器目录。

LeRobot ACT 文档：

https://huggingface.co/docs/lerobot/act

## 数据准备

数据集：

https://huggingface.co/datasets/huiwon/calvin_task_ABC_D

在 `topic2/` 目录运行：

```bash
python scripts/download_calvin.py
python scripts/convert_lerobot_v30.py
python scripts/prepare_data.py \
  --dataset-root data/lerobot_v21 \
  --output-dir data/splits
```

最终使用的数据目录：

```text
data/lerobot_v21/local/calvin_A
data/lerobot_v21/local/calvin_ABC
data/lerobot_v21/local/calvin_D
```

## 训练

```bash
python scripts/train.py --config configs/act_a_only.yaml
python scripts/train.py --config configs/act_abc.yaml
```

服务器可使用：

```bash
bash scripts/launch_best_training.sh
```

并行训练默认使用 GPU 0 和 GPU 1，可按机器资源覆盖：

```bash
GPU_A=0 GPU_ABC=7 bash scripts/launch_best_training.sh
```

训练包装器会：

- 每 1000 steps 评估训练域 validation split。
- 写入 `metrics/checkpoint_valid_metrics.csv`。
- 只维护 `checkpoints/best/pretrained_model`。
- 保留训练日志和可重新生成图表的 CSV。

## D 环境 Zero-shot 评估

```bash
python scripts/eval_zero_shot.py \
  --config configs/act_a_only.yaml \
  --checkpoint outputs/act_calvin_A/checkpoints/best/pretrained_model

python scripts/eval_zero_shot.py \
  --config configs/act_abc.yaml \
  --checkpoint outputs/act_calvin_ABC/checkpoints/best/pretrained_model
```

输出：

```text
outputs/<run>/metrics/eval_D_summary.csv
outputs/<run>/metrics/eval_D_episodes.csv
```

## 可视化

```bash
python scripts/parse_lerobot_log.py --run outputs/act_calvin_A
python scripts/parse_lerobot_log.py --run outputs/act_calvin_ABC

python scripts/plot_metrics.py \
  --runs outputs/act_calvin_A outputs/act_calvin_ABC \
  --output-dir outputs/figures
```

WandB 项目：

https://wandb.ai/zhanxing-fudan-university-school-of-management/CS60003-HW3-ACT

## 模型权重

Hugging Face：

```text
topic2/act_calvin_A_best/pretrained_model/
topic2/act_calvin_ABC_best/pretrained_model/
topic2/best_model_summary.json
```

https://huggingface.co/zhanxing/CS60003-HW3/tree/main/topic2
