# CS60003-HW3: 深度学习与空间智能

本仓库包含 HW3 两个题目的可复现实验代码、配置、轻量结果与 LaTeX 报告源文件。

- **题目一：基于 3DGS 与 AIGC 的多源资产生成与真实场景融合**
- **题目二：基于 LeRobot 的 ACT 策略跨环境泛化挑战**

大体积训练权重、导出模型与视频托管在 Hugging Face：

https://huggingface.co/zhanxing/CS60003-HW3/tree/main

## 仓库结构

```text
topic1/                 题目一代码、环境、配置与轻量结果
configs/                题目二 ACT 训练配置
scripts/                题目二数据准备、训练、评估与绘图脚本
src/hw3_act/            题目二核心实现
report_neurips/         两个题目的 Overleaf/XeLaTeX 报告
requirements.txt        题目二基础依赖
environment.yml         题目二 Conda 环境
```

## 题目一：3DGS 与 AIGC 多源资产融合

### 完成内容

1. 从手机环绕视频抽帧，通过 COLMAP 恢复相机位姿并训练物体 A 的 3DGS。
2. 使用 threestudio DreamFusion + Stable Diffusion v1.5 SDS 生成文本条件物体 B。
3. 对单张真实图像去背景，使用 Stable Zero123 生成物体 C。
4. 在 Mip-NeRF 360 Flowers 场景上训练背景 3DGS。
5. 将 AIGC mesh 表面采样为彩色点云，通过统一尺度、旋转和平移完成融合预览。

### 主要结果

| 模型 | 设置 | 结果 |
|---|---|---|
| 物体 A 3DGS | 49 个注册视角，7000 iterations | SSIM 0.9222 / PSNR 24.2401 / LPIPS 0.1431 |
| Flowers 背景 3DGS | 173 个视角，7000 iterations | SSIM 0.5455 / PSNR 20.8076 / LPIPS 0.4065 |
| 物体 B DreamFusion | 1200 steps | 1024 vertices / 2044 faces |
| 物体 C Stable Zero123 | 600 steps | 11848 vertices / 23692 faces |

### 环境与运行

题目一分别提供预览、3DGS 重建和 threestudio 三类环境文件：

```powershell
cd topic1
conda env create -f environment.preview.yml
conda env create -f environment.reconstruction.yml
powershell -ExecutionPolicy Bypass -File scripts/setup_threestudio_env.ps1
```

准备物体 A 与物体 C：

```powershell
python scripts/prepare_object_a.py --count 72
python scripts/prepare_object_c.py --method grabcut
python scripts/run_colmap_pipeline.py --workspace outputs/colmap/object_a_singlecam
```

训练 3DGS：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/train_object_a_3dgs.ps1 -Iterations 7000 -Resolution 2
powershell -ExecutionPolicy Bypass -File scripts/train_background_3dgs.ps1 -Iterations 7000 -Resolution 4
```

训练 AIGC 资产并生成融合预览：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_threestudio_object_b_dreamfusion_sd.ps1 -MaxSteps 1200
powershell -ExecutionPolicy Bypass -File scripts/run_threestudio_object_c_zero123.ps1 -MaxSteps 600
python scripts/generate_fusion_preview.py
```

更完整的环境说明、数据路径和导出命令见 [`topic1/README.md`](topic1/README.md)。

## 题目二：LeRobot ACT 跨环境泛化

题目二比较两个使用相同 ACT 架构与超参数的策略：

- `act_calvin_A`：仅使用 CALVIN 环境 A。
- `act_calvin_ABC`：联合使用环境 A/B/C。

模型只根据训练域 validation Action L1 选择最佳 checkpoint，环境 D 仅用于最终 zero-shot 评估。

### 最终设置与结果

- Training steps: `30000`
- Batch size: `64`
- Learning rate: `1e-4`
- Optimizer: `AdamW`
- ACT chunk size: `16`
- Validation: 每个 checkpoint 评估 `50` batches
- Final D evaluation: `200` batches

| 模型 | Best Step | Validation Action L1 | D Zero-shot Action L1 |
|---|---:|---:|---:|
| A-only | 14000 | 0.475544 | 0.496255 |
| A/B/C joint | 27000 | 0.427793 | 0.447098 |

D 的离线数据没有可靠任务完成信号，因此使用 Action L1 作为主要 zero-shot 指标。

### 数据准备

```bash
pip install -r requirements.txt
python scripts/download_calvin.py
python scripts/convert_lerobot_v30.py
python scripts/prepare_data.py --dataset-root data/lerobot_v21 --output-dir data/splits
```

数据集：https://huggingface.co/datasets/huiwon/calvin_task_ABC_D

LeRobot ACT 文档：https://huggingface.co/docs/lerobot/act

### 训练、评估与绘图

```bash
python scripts/train.py --config configs/act_a_only.yaml
python scripts/train.py --config configs/act_abc.yaml

python scripts/eval_zero_shot.py \
  --config configs/act_a_only.yaml \
  --checkpoint outputs/act_calvin_A/checkpoints/best/pretrained_model

python scripts/eval_zero_shot.py \
  --config configs/act_abc.yaml \
  --checkpoint outputs/act_calvin_ABC/checkpoints/best/pretrained_model

python scripts/plot_metrics.py \
  --runs outputs/act_calvin_A outputs/act_calvin_ABC \
  --output-dir outputs/figures
```

WandB 项目：

https://wandb.ai/zhanxing-fudan-university-school-of-management/CS60003-HW3-ACT

## 模型权重

Hugging Face 仓库包含：

```text
act_calvin_A_best/pretrained_model/
act_calvin_ABC_best/pretrained_model/
topic1/weights/
topic1/assets/
topic1/previews/
```

题目一权重包附带 SHA256 manifest；题目二保存 validation-best 的 LeRobot ACT checkpoint。

## 实验报告

`report_neurips/` 是可直接上传 Overleaf 的 XeLaTeX 项目，包含两个题目的完整内容、表格和报告使用图像。

本地编译：

```bash
cd report_neurips
xelatex main.tex
xelatex main.tex
```

请勿使用 pdfLaTeX 编译中文报告。
