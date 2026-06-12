# HW3 Task 2 Report Source

This directory is an Overleaf-ready LaTeX project for the report.

## Compile

1. Upload the whole `report_neurips/` directory to Overleaf.
2. Set compiler to **XeLaTeX**.
3. Compile `main.tex`.

The project uses a self-contained NeurIPS-like layout with `ctexart`, so it does not require downloading an external `.sty` file.

## TODO Before Submission

- Replace the name and student ID in `main.tex`.
- Confirm the GitHub repository URL:
  `https://github.com/Zhan-xing-pz/CS60003-HW3`
- Confirm the model weight URL:
  `https://huggingface.co/zhanxing/CS60003-HW3/tree/main`
- If the instructor strictly requires WandB/SwanLab exported images, upload the CSV files in `data/` to WandB/SwanLab or rerender the same curves there, then replace the note in Table 2 / Figure 1.

## Included Figures

- `figures/train_loss_curve.png`
- `figures/valid_action_l1_curve.png`
- `figures/eval_D_bar.png`
- `figures/action_l1_distribution.png`

## Included Metrics

- `data/act_calvin_A_checkpoint_valid_metrics.csv`
- `data/act_calvin_ABC_checkpoint_valid_metrics.csv`
- `data/act_calvin_A_eval_D_summary.csv`
- `data/act_calvin_ABC_eval_D_summary.csv`

## Template Note

Official NeurIPS style files are usually published on the NeurIPS conference site under Paper Information / Style Files. If strict NeurIPS formatting is required, create a new Overleaf project from the official NeurIPS template and copy the body of `main.tex` into that template.
