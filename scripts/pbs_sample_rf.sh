#!/bin/bash
#PBS -q eg
#PBS -l select=1:ngpus=1:nmigs=1
#PBS -l walltime=00:30:00
#PBS -W group_list=c30746
#PBS -N rf_sample
#PBS -j oe
#PBS -o /lustre1/work/c30636/test/subdiff/samples_rf_from_noise/pbs.out

set -eu

cd /lustre1/work/c30636/test/subdiff
mkdir -p samples_rf_from_noise

PY=/work/c30636/conda/envs/gdt/bin/python

$PY scripts/sample_flow.py \
  --config configs/pretrain_vit_b16_naive_rf.yaml \
  --checkpoint logs_naive_rf/checkpoints/checkpoint_best.pth \
  --num_samples 16 --num_steps 50 --batch_size 4 \
  --output_dir samples_rf_from_noise/
