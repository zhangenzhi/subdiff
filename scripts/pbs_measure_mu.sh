#!/bin/bash
#PBS -q eg
#PBS -l select=1:ngpus=1:nmigs=1
#PBS -l walltime=00:30:00
#PBS -W group_list=c30746
#PBS -N measure_mu
#PBS -j oe
#PBS -o /lustre1/work/c30636/test/subdiff/logs_dual_rf_p8/measure_mu.out

set -eu

cd /lustre1/work/c30636/test/subdiff

source /home/c30746/miniconda3/etc/profile.d/conda.sh
conda activate gdt

python scripts/measure_mu_x0_distance.py \
  --config configs/pretrain_vit_b8_dual_rf.yaml \
  --checkpoint logs_dual_rf_p8/checkpoints/checkpoint_latest.pth \
  --prompt_ratio 0.25 \
  --num_samples 512 \
  --batch_size 64 \
  --ts 1.0,0.75,0.5,0.25,0.0
