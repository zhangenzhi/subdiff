#!/bin/bash
#PBS -q eg
#PBS -l select=1:ngpus=1:nmigs=1
#PBS -l walltime=00:30:00
#PBS -W group_list=c30746
#PBS -N rf_dual_t1
#PBS -j oe
#PBS -o /lustre1/work/c30636/test/subdiff/samples_rf_dual_inpaint_t1/pbs.out

set -eu

cd /lustre1/work/c30636/test/subdiff
mkdir -p samples_rf_dual_inpaint_t1

PY=/work/c30636/conda/envs/gdt/bin/python

$PY scripts/inpaint_t1_grid.py \
  --config configs/pretrain_vit_b16_dual_rf.yaml \
  --checkpoint logs_dual_rf/checkpoints/checkpoint_latest.pth \
  --prompt_ratio 0.25 --num_samples 4 \
  --output_dir samples_rf_dual_inpaint_t1/
