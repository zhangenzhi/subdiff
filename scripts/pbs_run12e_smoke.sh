#!/bin/bash
#PBS -q c30636g
#PBS -l select=1:ngpus=2
#PBS -l walltime=00:30:00
#PBS -W group_list=c30636
#PBS -N run12e_smoke
#PBS -j oe
#PBS -o /lustre1/work/c30636/test/subdiff/logs_run12e_mae_p4_fp8/smoke.out

# 2-GPU smoke test for Run 12e Stage 1 (pure MAE p4 fp8). Runs ~1 epoch
# (or until job ends) and verifies:
#   - sdif env activates cleanly
#   - TE fp8_autocast works in DDP
#   - first-step compile + autocast does not OOM at batch=64 + p4
#   - per-step time falls into expected range (~390ms based on probe)
#   - loss decreases sanely
#   - checkpoint saves with clean state_dict keys (no _orig_mod prefix etc)

set -eu

cd /lustre1/work/c30636/test/subdiff
mkdir -p logs_run12e_mae_p4_fp8

source /home/c30746/miniconda3/etc/profile.d/conda.sh
conda activate sdif
module load cuda/12.8.0
export CUDA_HOME=/work/opt/local/x86_64/cores/cuda/12.8.0

NGPUS=2
MASTER_PORT=29502

torchrun \
  --nnodes=1 \
  --nproc_per_node=$NGPUS \
  --master_port=$MASTER_PORT \
  scripts/pretrain_mae_te.py \
  --config configs/pretrain_mae_p4_fp8.yaml
