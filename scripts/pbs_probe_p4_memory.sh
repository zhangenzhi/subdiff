#!/bin/bash
#PBS -q c30636g
#PBS -l select=1:ngpus=1
#PBS -l walltime=00:15:00
#PBS -W group_list=c30636
#PBS -N probe_p4_mem
#PBS -j oe
#PBS -o /lustre1/work/c30636/test/subdiff/logs_run12_cold_rf/probe_p4_mem.out

set -eu

cd /lustre1/work/c30636/test/subdiff
mkdir -p logs_run12_cold_rf

source /home/c30746/miniconda3/etc/profile.d/conda.sh
conda activate gdt

python scripts/probe_p4_memory.py
