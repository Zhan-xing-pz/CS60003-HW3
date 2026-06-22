#!/usr/bin/env bash
set -euo pipefail

cd /data2/pz/homework/hw3/topic2

tmux kill-session -t hw3_act_A_best 2>/dev/null || true
tmux kill-session -t hw3_act_ABC_best 2>/dev/null || true

tmux new-session -d -s hw3_act_A_best \
  "bash -lc 'cd /data2/pz/homework/hw3/topic2 && mkdir -p outputs && exec > outputs/act_calvin_A_tmux_boot.log 2>&1 && set -x && export CUDA_VISIBLE_DEVICES=0 && export HF_ENDPOINT=https://hf-mirror.com && /data/anaconda3/envs/pz/bin/python scripts/train.py --config configs/act_a_only.yaml'"

tmux new-session -d -s hw3_act_ABC_best \
  "bash -lc 'cd /data2/pz/homework/hw3/topic2 && mkdir -p outputs && exec > outputs/act_calvin_ABC_tmux_boot.log 2>&1 && set -x && export CUDA_VISIBLE_DEVICES=7 && export HF_ENDPOINT=https://hf-mirror.com && /data/anaconda3/envs/pz/bin/python scripts/train.py --config configs/act_abc.yaml'"

tmux ls | grep 'hw3_act'
