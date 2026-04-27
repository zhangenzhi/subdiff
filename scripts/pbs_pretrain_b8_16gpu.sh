#!/bin/bash
#PBS -q lg
#PBS -l select=4:ngpus=4
#PBS -l walltime=24:00:00
#PBS -W group_list=c30746
#PBS -N rf_dual_p8_16gpu
#PBS -j oe
#PBS -o /lustre1/work/c30636/test/subdiff/logs_dual_rf_p8/pbs.out

# Multi-node DDP via torchrun rendezvous.
# 4 chunks × 4 GPU each = 16 GPUs across 4 sgpu nodes.
#
# NOTE: prior 2-chunk submission was rejected by the queue. If this 4-chunk
# request is also rejected, fall back to single-node 4-GPU + longer training.

set -eu

cd /lustre1/work/c30636/test/subdiff
mkdir -p logs_dual_rf_p8

# Distinct node list
NODES=$(sort -u "$PBS_NODEFILE")
NNODES=$(echo "$NODES" | wc -l)
NGPUS_PER_NODE=4
MASTER_ADDR=$(echo "$NODES" | head -1)
MASTER_PORT=29500

PY=/work/c30636/conda/envs/gdt/bin/python
TORCHRUN=/work/c30636/conda/envs/gdt/bin/torchrun

echo "=== Multi-node setup ==="
echo "PBS_JOBID=$PBS_JOBID"
echo "Nodes: $NODES"
echo "NNODES=$NNODES NGPUS_PER_NODE=$NGPUS_PER_NODE"
echo "MASTER=$MASTER_ADDR:$MASTER_PORT"

# Launch torchrun on each node. The first node runs locally; others via
# pbs_tmrsh (uses PBS Task Manager, no user-level ssh keys needed).
# We delegate to scripts/_torchrun_node.sh so each call is a clean
# argv list (no quoting hell).
LAUNCHER=/lustre1/work/c30636/test/subdiff/scripts/_torchrun_node.sh
CONFIG=configs/pretrain_vit_b8_dual_rf.yaml

RANK=0
PIDS=()
for NODE in $NODES; do
  if [ "$NODE" = "$(hostname -s)" ] || [ "$NODE" = "$(hostname)" ]; then
    "$LAUNCHER" "$RANK" "$NNODES" "$MASTER_ADDR" "$CONFIG" &
  else
    pbs_tmrsh "$NODE" "$LAUNCHER" "$RANK" "$NNODES" "$MASTER_ADDR" "$CONFIG" &
  fi
  PIDS+=($!)
  RANK=$((RANK + 1))
done

# Wait for all to finish; propagate first non-zero exit
EXIT=0
for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then
    EXIT=$?
  fi
done
exit $EXIT
