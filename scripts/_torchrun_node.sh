#!/bin/bash
# Per-node torchrun launcher invoked by pbs_tmrsh.
# Args: $1 = node_rank
#       $2 = nnodes
#       $3 = master_addr
#       $4 = config path (relative to repo root)

set -eu

NODE_RANK=$1
NNODES=$2
MASTER_ADDR=$3
CONFIG=$4
NGPUS_PER_NODE=4
MASTER_PORT=29500

cd /lustre1/work/c30636/test/subdiff

# Activate conda gdt env (works on every node without absolute paths)
source /home/c30746/miniconda3/etc/profile.d/conda.sh
conda activate gdt

echo "[$(hostname)] node_rank=$NODE_RANK nnodes=$NNODES master=$MASTER_ADDR cfg=$CONFIG"
echo "[$(hostname)] python=$(which python)  torchrun=$(which torchrun)"

exec torchrun \
  --nnodes=$NNODES \
  --nproc_per_node=$NGPUS_PER_NODE \
  --node_rank=$NODE_RANK \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  scripts/pretrain.py --config "$CONFIG"
