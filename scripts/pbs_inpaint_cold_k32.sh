#!/bin/bash
#PBS -q eg
#PBS -l select=1:ngpus=1:nmigs=1
#PBS -l walltime=00:30:00
#PBS -W group_list=c30746
#PBS -N run12_k32
#PBS -j oe
#PBS -o /lustre1/work/c30636/test/subdiff/samples_run12_inpaint/k32.out

# Clean 4-column comparison for documentation:
#   Original | Clean prompt | Run X μ (single) | Run 12 K=32
# K=32 is the sweet-spot per the K=1..64 sweep.

set -eu

cd /lustre1/work/c30636/test/subdiff
mkdir -p samples_run12_inpaint

source /home/c30746/miniconda3/etc/profile.d/conda.sh
conda activate gdt

python scripts/inpaint_cold_dual.py \
  --mu_config configs/pretrain_vit_b8_dual_rf.yaml \
  --mu_ckpt   logs_dual_rf_p8/checkpoints/checkpoint_latest.pth \
  --refiner_config configs/pretrain_vit_b8_cold_rf.yaml \
  --refiner_ckpt   logs_run12_cold_rf/checkpoints/checkpoint_latest.pth \
  --Ks 32 \
  --num_samples 4 \
  --output_dir samples_run12_inpaint/

EP=$(python -c "import torch; c=torch.load('logs_run12_cold_rf/checkpoints/checkpoint_latest.pth', map_location='cpu', weights_only=False); print(c['epoch'])")
mv samples_run12_inpaint/cold_t1_p0.25.png samples_run12_inpaint/k32_clean_ep${EP}.png
echo "Saved as k32_clean_ep${EP}.png"
