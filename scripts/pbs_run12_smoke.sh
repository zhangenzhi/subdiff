#!/bin/bash
#PBS -q eg
#PBS -l select=1:ngpus=1:nmigs=1
#PBS -l walltime=00:30:00
#PBS -W group_list=c30746
#PBS -N run12_smoke
#PBS -j oe
#PBS -o /lustre1/work/c30636/test/subdiff/logs_run12_cold_rf/smoke.out

# Single-GPU smoke test: load Refiner + frozen mu_model, run 5 forward+
# backward steps on batch=4. Verifies the cold path end-to-end before
# committing 16 GPU.

set -eu

cd /lustre1/work/c30636/test/subdiff
mkdir -p logs_run12_cold_rf

source /home/c30746/miniconda3/etc/profile.d/conda.sh
conda activate gdt

python - <<'PY'
import os, sys, yaml, torch
sys.path.insert(0, '/lustre1/work/c30636/test/subdiff')
from scripts.pretrain import build_subdiff_from_cfg

device = torch.device('cuda')
torch.manual_seed(0)

with open('configs/pretrain_vit_b8_cold_rf.yaml') as f:
    cfg = yaml.safe_load(f)

curriculum_cfg = {k: cfg['curriculum'][k] for k in
    ['t_min_start','t_min_end','t_max_start','t_max_end',
     'clean_ratio_start','clean_ratio_end','warmup_epochs','schedule']}

# Build refiner
refiner = build_subdiff_from_cfg(cfg, curriculum_cfg).to(device)
print(f'Refiner params: {sum(p.numel() for p in refiner.parameters())/1e6:.1f}M, cold_rf={refiner.cold_rf}')

# Build mu_model from Run X config + ckpt
with open(cfg['model']['cold_rf_mu_config']) as f:
    mu_cfg = yaml.safe_load(f)
mu_curriculum = {k: mu_cfg['curriculum'][k] for k in
    ['t_min_start','t_min_end','t_max_start','t_max_end',
     'clean_ratio_start','clean_ratio_end','warmup_epochs','schedule']}
mu_model = build_subdiff_from_cfg(mu_cfg, mu_curriculum).to(device)
ckpt = torch.load(cfg['model']['cold_rf_mu_ckpt'], map_location=device, weights_only=False)
state = ckpt['model']
if getattr(mu_model.encoder, 'pos_embed_type', 'learnable') == 'sincos':
    state = {k:v for k,v in state.items() if not k.endswith('encoder.pos_embed')}
missing, unexpected = mu_model.load_state_dict(state, strict=False)
print(f'mu_model loaded ep={ckpt["epoch"]} avg_loss={ckpt.get("avg_loss"):.4f}')
print(f'  missing={len(missing)} unexpected={len(unexpected)}')
mu_model.eval()
for p in mu_model.parameters(): p.requires_grad = False

# Smoke-test 5 steps
opt = torch.optim.AdamW(refiner.parameters(), lr=1e-4)
imgs = torch.randn(4, 3, 224, 224, device=device)
for step in range(5):
    B = imgs.shape[0]
    N = refiner.num_patches
    noisy_mask = refiner.diffusion.generate_noisy_mask(B, N, refiner.clean_ratio, device)
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        mu = mu_model.compute_mu(imgs, noisy_mask)
    refiner.set_cold_context(mu, noisy_mask)
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        loss, log = refiner(imgs, epoch=0)
    opt.zero_grad()
    loss.backward()
    opt.step()
    print(f'  step {step}: loss={loss.item():.5f}  '
          f't_mean={log["t_mean"].item():.3f}  '
          f'mu_norm={(mu**2).mean().item():.4f}')

print('SMOKE TEST PASS')
PY
