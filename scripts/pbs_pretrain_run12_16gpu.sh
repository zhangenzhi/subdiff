#!/bin/bash
#PBS -q lg
#PBS -l select=4:ngpus=4
#PBS -l walltime=24:00:00
#PBS -W group_list=c30746
#PBS -N run12_cold_rf_16gpu
#PBS -j oe
#PBS -o /lustre1/work/c30636/test/subdiff/logs_run12_cold_rf/pbs.out

# Run 12 — Cold-RF Refiner training on 16 GPU (4 sgpu nodes × 4 GPU).
# Same multi-node launch as Run X (pbs_pretrain_b8_16gpu.sh), only the
# config path differs.

set -eu

cd /lustre1/work/c30636/test/subdiff
mkdir -p logs_run12_cold_rf

NODES=$(sort -u "$PBS_NODEFILE")
NNODES=$(echo "$NODES" | wc -l)
NGPUS_PER_NODE=4
MASTER_ADDR=$(echo "$NODES" | head -1)
MASTER_PORT=29500

echo "=== Multi-node setup ==="
echo "PBS_JOBID=$PBS_JOBID"
echo "Nodes: $NODES"
echo "NNODES=$NNODES NGPUS_PER_NODE=$NGPUS_PER_NODE"
echo "MASTER=$MASTER_ADDR:$MASTER_PORT"

LAUNCHER=/lustre1/work/c30636/test/subdiff/scripts/_torchrun_node.sh
CONFIG=configs/pretrain_vit_b8_cold_rf.yaml

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

EXIT=0
for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then
    EXIT=$?
  fi
done
exit $EXIT
