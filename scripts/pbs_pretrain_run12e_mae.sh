#!/bin/bash
#PBS -q lg
#PBS -l select=4:ngpus=4
#PBS -l walltime=24:00:00
#PBS -W group_list=c30746
#PBS -N run12e_mae
#PBS -j oe
#PBS -o /lustre1/work/c30636/test/subdiff/logs_run12e_mae_p4_fp8/pbs.out

# Run 12e Stage 1 — classical asymmetric MAE μ-generator at p4 + fp8.
# 4 nodes × 4 GPU = 16 H100, batch=64/GPU = 1024 global, 100 epochs.
# Predicted wall: ~11h based on smoke step time of 326ms × 1250 steps/epoch.
#
# Resume support: chain via `qsub -W depend=afterok:<this_id>`.

set -eu

cd /lustre1/work/c30636/test/subdiff
mkdir -p logs_run12e_mae_p4_fp8

NODES=$(sort -u "$PBS_NODEFILE")
NNODES=$(echo "$NODES" | wc -l)
NGPUS_PER_NODE=4
MASTER_ADDR=$(echo "$NODES" | head -1)

echo "=== Multi-node setup ==="
echo "PBS_JOBID=$PBS_JOBID"
echo "Nodes: $NODES"
echo "NNODES=$NNODES NGPUS_PER_NODE=$NGPUS_PER_NODE"
echo "MASTER=$MASTER_ADDR"

LAUNCHER=/lustre1/work/c30636/test/subdiff/scripts/_torchrun_node_mae_te.sh
CONFIG=configs/pretrain_mae_p4_fp8.yaml
RESUME=${RESUME:-}

if [ -n "$RESUME" ] && [ ! -f "$RESUME" ]; then
  echo "FATAL: RESUME=$RESUME requested but file not found"
  exit 1
fi

RANK=0
PIDS=()
for NODE in $NODES; do
  if [ "$NODE" = "$(hostname -s)" ] || [ "$NODE" = "$(hostname)" ]; then
    "$LAUNCHER" "$RANK" "$NNODES" "$MASTER_ADDR" "$CONFIG" "$RESUME" &
  else
    pbs_tmrsh "$NODE" "$LAUNCHER" "$RANK" "$NNODES" "$MASTER_ADDR" "$CONFIG" "$RESUME" &
  fi
  PIDS+=($!)
  RANK=$((RANK + 1))
done

EXIT=0
for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then
    EXIT=$?
  fi
done
exit $EXIT
