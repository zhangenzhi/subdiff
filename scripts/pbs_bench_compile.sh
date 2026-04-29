#!/bin/bash
#PBS -q c30636g
#PBS -l select=1:ngpus=2
#PBS -l walltime=00:30:00
#PBS -W group_list=c30636
#PBS -N bench_compile
#PBS -j oe
#PBS -o /lustre1/work/c30636/test/subdiff/logs_run12_cold_rf/bench_compile.out

# 2-GPU DDP benchmark across the (compile mode) × (SDPA backend) grid.
# 4 runs total — same node, same process startup cost, only one variable
# changes between rows.

set -eu

cd /lustre1/work/c30636/test/subdiff
mkdir -p logs_run12_cold_rf

source /home/c30746/miniconda3/etc/profile.d/conda.sh
conda activate gdt

NGPUS=2
MASTER_PORT=29501
RUN() {
  local label=$1; shift
  echo
  echo "============================================================"
  echo "===== $label ====="
  echo "============================================================"
  "$@" torchrun \
    --nnodes=1 \
    --nproc_per_node=$NGPUS \
    --master_port=$MASTER_PORT \
    scripts/bench_compile.py
}

RUN "Run 1/4: eager + flash (FA2)"               env COMPILE=0 SDPA=flash
RUN "Run 2/4: eager + cudnn (FA3)"               env COMPILE=0 SDPA=cudnn
RUN "Run 3/4: compile=default + flash (FA2)"     env COMPILE=1 COMPILE_MODE=default SDPA=flash
RUN "Run 4/4: compile=default + cudnn (FA3)"     env COMPILE=1 COMPILE_MODE=default SDPA=cudnn

echo
echo "===== bench_compile DONE ====="
