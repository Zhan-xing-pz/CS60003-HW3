# CS60003-HW3: 深度学习与空间智能

本项目按作业题目划分为两个独立子目录，公共报告与仓库说明保留在根目录。

```text
topic1/                 题目一：3DGS 与 AIGC 多源 3D 资产融合
topic2/                 题目二：LeRobot ACT 跨环境泛化
report_neurips/         两个题目的统一 LaTeX/PDF 实验报告
README.md               项目总览
```

## 题目一

题目一包含：

- COLMAP + 3DGS 真实物体重建
- DreamFusion 文本到 3D
- Stable Zero123 单图到 3D
- Mip-NeRF 360 Flowers 背景 3DGS
- mesh/point cloud 统一表示与融合预览

代码与运行说明见 [`topic1/README.md`](topic1/README.md)。

## 题目二

题目二包含：

- CALVIN A-only ACT 训练
- CALVIN A/B/C 联合 ACT 训练
- 训练域 validation-best checkpoint 选择
- 环境 D zero-shot Action L1 评估
- WandB、CSV 与本地图表

代码与运行说明见 [`topic2/README.md`](topic2/README.md)。

## 模型与资产

Hugging Face 采用与 GitHub 对称的目录：

```text
topic1/weights/
topic1/assets/
topic1/previews/
topic2/act_calvin_A_best/
topic2/act_calvin_ABC_best/
topic2/best_model_summary.json
```

模型仓库：

https://huggingface.co/zhanxing/CS60003-HW3/tree/main

## 实验报告

`report_neurips/` 是完整的 Overleaf/XeLaTeX 项目，覆盖题目一和题目二。

```bash
cd report_neurips
xelatex main.tex
xelatex main.tex
```

GitHub：

https://github.com/Zhan-xing-pz/CS60003-HW3

WandB：

https://wandb.ai/zhanxing-fudan-university-school-of-management/CS60003-HW3-ACT
