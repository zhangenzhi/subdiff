#!/bin/bash
#PBS -q lg
#PBS -l select=4:ngpus=4
#PBS -l walltime=24:00:00
#PBS -W group_list=c30746
#PBS -N run12_resume
#PBS -j oe
#PBS -o /lustre1/work/c30636/test/subdiff/logs_run12_cold_rf/pbs_resume.out

# Resume Run 12 (cold-RF) from checkpoint_latest.pth and continue toward the
# config's `training.epochs` (currently 400). One 24h job covers ~232 epochs
# at 6.2 min/ep, so going from ep 99 → 400 takes ~2 chained jobs.
#
# To chain: submit this with `qsub -W depend=afterok:<this_job_id>`.

set -eu

cd /lustre1/work/c30636/test/subdiff
mkdir -p logs_run12_cold_rf

NODES=$(sort -u "$PBS_NODEFILE")
NNODES=$(echo "$NODES" | wc -l)
NGPUS_PER_NODE=4
MASTER_ADDR=$(echo "$NODES" | head -1)

echo "=== Multi-node setup ==="
echo "PBS_JOBID=$PBS_JOBID"
echo "Nodes: $NODES"
echo "NNODES=$NNODES NGPUS_PER_NODE=$NGPUS_PER_NODE"
echo "MASTER=$MASTER_ADDR"

LAUNCHER=/lustre1/work/c30636/test/subdiff/scripts/_torchrun_node.sh
CONFIG=configs/pretrain_vit_b8_cold_rf.yaml
RESUME=logs_run12_cold_rf/checkpoints/checkpoint_latest.pth

if [ ! -f "$RESUME" ]; then
  echo "FATAL: resume checkpoint not found at $RESUME"
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
