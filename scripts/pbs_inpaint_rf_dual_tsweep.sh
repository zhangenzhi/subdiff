#!/bin/bash
#PBS -q eg
#PBS -l select=1:ngpus=1:nmigs=1
#PBS -l walltime=00:30:00
#PBS -W group_list=c30746
#PBS -N rf_dual_tsweep
#PBS -j oe
#PBS -o /lustre1/work/c30636/test/subdiff/samples_rf_dual_inpaint/pbs_tsweep.out

set -eu

cd /lustre1/work/c30636/test/subdiff
mkdir -p samples_rf_dual_inpaint

PY=/work/c30636/conda/envs/gdt/bin/python

# t-sweep at prompt=0% — push toward pure noise to find collapse
$PY scripts/inpaint_rf_dual.py \
  --config configs/pretrain_vit_b16_dual_rf.yaml \
  --checkpoint logs_dual_rf/checkpoints/checkpoint_latest.pth \
  --prompt_ratios 0.0 \
  --t_vals 0.5 0.7 0.85 0.95 0.99 \
  --num_samples 4 \
  --output_dir samples_rf_dual_inpaint/

# Same t-sweep but with prompt=25% — to see if the prompt rescues high-t inputs
$PY scripts/inpaint_rf_dual.py \
  --config configs/pretrain_vit_b16_dual_rf.yaml \
  --checkpoint logs_dual_rf/checkpoints/checkpoint_latest.pth \
  --prompt_ratios 0.25 \
  --t_vals 0.5 0.7 0.85 0.95 0.99 \
  --num_samples 4 \
  --output_dir samples_rf_dual_inpaint/
