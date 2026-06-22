#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${HW3_PYTHON:-$(command -v python)}"
GPU_A="${GPU_A:-0}"
GPU_ABC="${GPU_ABC:-1}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

tmux kill-session -t hw3_act_A_best 2>/dev/null || true
tmux kill-session -t hw3_act_ABC_best 2>/dev/null || true

tmux new-session -d -s hw3_act_A_best \
  "cd '${PROJECT_ROOT}' && mkdir -p outputs && exec > outputs/act_calvin_A_tmux_boot.log 2>&1 && set -x && export CUDA_VISIBLE_DEVICES='${GPU_A}' HF_ENDPOINT='${HF_ENDPOINT}' && '${PYTHON_BIN}' scripts/train.py --config configs/act_a_only.yaml"

tmux new-session -d -s hw3_act_ABC_best \
  "cd '${PROJECT_ROOT}' && mkdir -p outputs && exec > outputs/act_calvin_ABC_tmux_boot.log 2>&1 && set -x && export CUDA_VISIBLE_DEVICES='${GPU_ABC}' HF_ENDPOINT='${HF_ENDPOINT}' && '${PYTHON_BIN}' scripts/train.py --config configs/act_abc.yaml"

tmux ls | grep 'hw3_act'
