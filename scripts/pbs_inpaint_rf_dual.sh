#!/bin/bash
#PBS -q eg
#PBS -l select=1:ngpus=1:nmigs=1
#PBS -l walltime=00:30:00
#PBS -W group_list=c30746
#PBS -N rf_dual_inpaint
#PBS -j oe
#PBS -o /lustre1/work/c30636/test/subdiff/samples_rf_dual_inpaint/pbs.out

set -eu

cd /lustre1/work/c30636/test/subdiff
mkdir -p samples_rf_dual_inpaint

PY=/work/c30636/conda/envs/gdt/bin/python

$PY scripts/inpaint_rf_dual.py \
  --config configs/pretrain_vit_b16_dual_rf.yaml \
  --checkpoint logs_dual_rf/checkpoints/checkpoint_latest.pth \
  --prompt_ratios 0.25 0.15 0.05 0.0 \
  --t_val 0.5 --num_samples 4 \
  --output_dir samples_rf_dual_inpaint/
