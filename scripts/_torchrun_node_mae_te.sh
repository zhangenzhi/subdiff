#!/bin/bash
# Per-node torchrun launcher for Run 12e Stage 1 (MAE + fp8 in sdif env).
# Args: $1 = node_rank
#       $2 = nnodes
#       $3 = master_addr
#       $4 = config path
#       $5 = (optional) resume checkpoint path → adds --resume <path>

set -eu

NODE_RANK=$1
NNODES=$2
MASTER_ADDR=$3
CONFIG=$4
RESUME=${5:-}
NGPUS_PER_NODE=4
MASTER_PORT=29502

EXTRA_ARGS=""
if [ -n "$RESUME" ]; then
  EXTRA_ARGS="--resume $RESUME"
fi

cd /lustre1/work/c30636/test/subdiff

# sdif env (clone of gdt + TransformerEngine for fp8)
source /home/c30746/miniconda3/etc/profile.d/conda.sh
conda activate sdif
module load cuda/12.8.0
export CUDA_HOME=/work/opt/local/x86_64/cores/cuda/12.8.0

echo "[$(hostname)] node_rank=$NODE_RANK nnodes=$NNODES master=$MASTER_ADDR cfg=$CONFIG resume='$RESUME'"
echo "[$(hostname)] python=$(which python)  torchrun=$(which torchrun)  cuda=$CUDA_HOME"

exec torchrun \
  --nnodes=$NNODES \
  --nproc_per_node=$NGPUS_PER_NODE \
  --node_rank=$NODE_RANK \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  scripts/pretrain_mae_te.py --config "$CONFIG" $EXTRA_ARGS
