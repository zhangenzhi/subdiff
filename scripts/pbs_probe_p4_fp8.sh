#!/bin/bash
#PBS -q c30636g
#PBS -l select=1:ngpus=1
#PBS -l walltime=00:20:00
#PBS -W group_list=c30636
#PBS -N probe_p4_fp8
#PBS -j oe
#PBS -o /lustre1/work/c30636/test/subdiff/logs_run12_cold_rf/probe_p4_fp8.out

set -eu

cd /lustre1/work/c30636/test/subdiff
mkdir -p logs_run12_cold_rf

# Use the new sdif env (clone of gdt + TransformerEngine for fp8)
source /home/c30746/miniconda3/etc/profile.d/conda.sh
conda activate sdif
module load cuda/12.8.0
export CUDA_HOME=/work/opt/local/x86_64/cores/cuda/12.8.0

python scripts/probe_p4_fp8.py
