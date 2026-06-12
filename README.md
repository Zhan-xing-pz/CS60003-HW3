# CS60003-HW3: LeRobot ACT Cross-Environment Generalization

This repository contains the code for HW3 Task 2, **基于 LeRobot 的 ACT 策略跨环境泛化挑战**.

The experiment compares two ACT policies on CALVIN-style LeRobot data:

- `act_calvin_A`: trained only on environment A.
- `act_calvin_ABC`: trained jointly on environments A/B/C.

Both policies use the same ACT architecture and hyperparameters. Checkpoints are selected by validation Action L1 on training-domain validation splits. Environment D is used only for final zero-shot evaluation.

## Final Setting

The final run uses:

- Training steps: `30000`
- Batch size: `64`
- Learning rate: `1e-4`
- Optimizer: `AdamW`
- Weight decay: `1e-6`
- Loss: L1 imitation loss
- ACT chunk size: `16`
- Validation per checkpoint: `50` batches
- Final D evaluation: `200` batches

Final validation-best results:

| Model | Best Step | Validation Action L1 | D Zero-shot Action L1 |
|---|---:|---:|---:|
| A-only | 14000 | 0.475544 | 0.496255 |
| A/B/C joint | 27000 | 0.427793 | 0.447098 |

The D split does not provide a reliable task success signal in the converted offline data, so the report uses Action L1 as the main zero-shot metric.

## Repository Layout

```text
configs/                Training/evaluation configs
scripts/                Data preparation, training, evaluation, plotting scripts
src/hw3_act/            Core implementation
report_neurips/         Overleaf-ready LaTeX report source and lightweight figures
requirements.txt        Python package requirements
environment.yml         Conda environment skeleton
```

Large files are intentionally excluded from Git:

- converted datasets
- training outputs
- checkpoints / model weights
- temporary files
- WandB/SwanLab caches

## Environment

The full experiment was run on a remote GPU server with:

```bash
conda activate pz
```

For a fresh environment, install the basic dependencies:

```bash
pip install -r requirements.txt
```

Install LeRobot following the official documentation if it is not already available:

```text
https://huggingface.co/docs/lerobot
```

## Data Preparation

The experiment uses the CALVIN A/B/C/D dataset:

```text
https://huggingface.co/datasets/huiwon/calvin_task_ABC_D
```

Typical preparation workflow:

```bash
python scripts/download_calvin.py
python scripts/convert_lerobot_v30.py
python scripts/prepare_data.py --dataset-root data/lerobot_v21 --output-dir data/splits
```

The final training used LeRobot v3.0-compatible local datasets:

```text
data/lerobot_v21/local/calvin_A
data/lerobot_v21/local/calvin_ABC
data/lerobot_v21/local/calvin_D
```

## Training

Train the A-only policy:

```bash
python scripts/train.py --config configs/act_a_only.yaml
```

Train the A/B/C joint policy:

```bash
python scripts/train.py --config configs/act_abc.yaml
```

On the server, both runs can be launched in tmux:

```bash
bash scripts/launch_best_training.sh
```

The training wrapper:

- evaluates every saved checkpoint on the training-domain validation split
- records `metrics/checkpoint_valid_metrics.csv`
- keeps only `checkpoints/best/pretrained_model`
- saves `metrics/train_metrics.csv` after log parsing

## Zero-Shot Evaluation on D

Evaluate validation-best checkpoints:

```bash
python scripts/eval_zero_shot.py \
  --config configs/act_a_only.yaml \
  --checkpoint outputs/act_calvin_A/checkpoints/best/pretrained_model

python scripts/eval_zero_shot.py \
  --config configs/act_abc.yaml \
  --checkpoint outputs/act_calvin_ABC/checkpoints/best/pretrained_model
```

Outputs:

```text
outputs/<run>/metrics/eval_D_summary.csv
outputs/<run>/metrics/eval_D_episodes.csv
```

## Plotting

Parse LeRobot logs and generate figures:

```bash
python scripts/parse_lerobot_log.py --run outputs/act_calvin_A
python scripts/parse_lerobot_log.py --run outputs/act_calvin_ABC

python scripts/plot_metrics.py \
  --runs outputs/act_calvin_A outputs/act_calvin_ABC \
  --output-dir outputs/figures
```

The report source in `report_neurips/` includes lightweight exported figures and CSV metrics for reproducibility.

## Model Weights

The validation-best LeRobot ACT checkpoints are hosted on Hugging Face:

```text
https://huggingface.co/zhanxing/CS60003-HW3/tree/main
```

Uploaded paths:

```text
act_calvin_A_best/pretrained_model/
act_calvin_ABC_best/pretrained_model/
best_model_summary.json
```

## Notes

- WandB was disabled in the final run because the server root partition was nearly full and WandB artifact staging writes to the home directory.
- Local CSV/PNG evidence is always saved, so plots and report tables can be regenerated without rerunning training.
- Model weights are stored separately from GitHub to avoid committing large binaries.
