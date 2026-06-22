# HW3 Report Source

This directory is an Overleaf-ready LaTeX project for the complete HW3 report:

- Task 1: 3DGS and AIGC multi-source 3D asset generation and scene fusion.
- Task 2: LeRobot ACT cross-environment generalization on CALVIN.

## Compile

1. Upload the whole `report_neurips/` directory to Overleaf.
2. Set compiler to **XeLaTeX**. The included `latexmkrc` also forces XeLaTeX for Overleaf/latexmk.
3. Compile `main.tex`.

The project uses a self-contained NeurIPS-like layout with `ctexart`, so it does not require downloading an external `.sty` file.
Do not compile this report with pdfLaTeX; Chinese `ctexart` fontsets such as Fandol require XeLaTeX or LuaLaTeX.

## Before Submission

- Confirm the GitHub repository URL:
  `https://github.com/Zhan-xing-pz/CS60003-HW3`
- Confirm the model weight URL:
  `https://huggingface.co/zhanxing/CS60003-HW3/tree/main`
- Confirm the WandB visualization project:
  `https://wandb.ai/zhanxing-fudan-university-school-of-management/CS60003-HW3-ACT`

## Included Figures

- `figures/topic1_asset_overview.jpg`
- `figures/topic1_object_a_colmap_quality.png`
- `figures/topic1_object_a_training_metrics.png`
- `figures/topic1_object_a_eval_grid.jpg`
- `figures/topic1_background_training_metrics.png`
- `figures/topic1_background_eval_grid.jpg`
- `figures/topic1_object_b_turntable.jpg`
- `figures/topic1_object_c_turntable.jpg`
- `figures/topic1_fusion_preview.jpg`
- `figures/topic1_method_comparison.png`
- `figures/wandb_export_train_loss.png` (used in the report)
- `figures/wandb_export_valid_action_l1.png` (used in the report)
- `figures/eval_D_bar.png`
- `figures/action_l1_distribution.png`

## Included Metrics

- `data/act_calvin_A_train_metrics.csv`
- `data/act_calvin_ABC_train_metrics.csv`
- `data/act_calvin_A_wandb_history.csv`
- `data/act_calvin_ABC_wandb_history.csv`
- `data/act_calvin_A_checkpoint_valid_metrics.csv`
- `data/act_calvin_ABC_checkpoint_valid_metrics.csv`
- `data/act_calvin_A_eval_D_summary.csv`
- `data/act_calvin_ABC_eval_D_summary.csv`
- `data/wandb_runs.json`

## Template Note

Official NeurIPS style files are usually published on the NeurIPS conference site under Paper Information / Style Files. If strict NeurIPS formatting is required, create a new Overleaf project from the official NeurIPS template and copy the body of `main.tex` into that template.
