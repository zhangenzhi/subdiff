#!/bin/bash
#PBS -q eg
#PBS -l select=1:ngpus=1:nmigs=1
#PBS -l walltime=00:30:00
#PBS -W group_list=c30746
#PBS -N compile_smoke
#PBS -j oe
#PBS -o /lustre1/work/c30636/test/subdiff/logs_run12_cold_rf/compile_smoke.out

# Single-GPU smoke for torch.compile + FA3 changes:
#   - verify HPC status print shows flash_sdp=True on H100/sm_90
#   - run 12 steps, print per-step time → first slow (compile trace),
#     steady-state should be < non-compile baseline
#   - save a checkpoint, reload it into a fresh non-compiled model
#     to confirm state_dict key format unchanged (no _orig_mod. prefix)

set -eu

cd /lustre1/work/c30636/test/subdiff
mkdir -p logs_run12_cold_rf

source /home/c30746/miniconda3/etc/profile.d/conda.sh
conda activate gdt

python - <<'PY'
import os, sys, time, yaml, torch
sys.path.insert(0, '/lustre1/work/c30636/test/subdiff')
from scripts.pretrain import (build_subdiff_from_cfg,
                               _enable_hpc_speedups, _print_hpc_status)

_enable_hpc_speedups()
_print_hpc_status(is_main=True)

device = torch.device('cuda')
torch.manual_seed(0)

with open('configs/pretrain_vit_b8_cold_rf.yaml') as f:
    cfg = yaml.safe_load(f)

curriculum_cfg = {k: cfg['curriculum'][k] for k in
    ['t_min_start','t_min_end','t_max_start','t_max_end',
     'clean_ratio_start','clean_ratio_end','warmup_epochs','schedule']}

refiner = build_subdiff_from_cfg(cfg, curriculum_cfg).to(device)
print(f'[smoke] Refiner: {sum(p.numel() for p in refiner.parameters())/1e6:.1f}M, '
      f'cold_rf={refiner.cold_rf}')

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
mu_model.load_state_dict(state, strict=False)
mu_model.eval()
for p in mu_model.parameters(): p.requires_grad = False

# ---------- Compile ----------
mode = cfg['training'].get('compile_mode', 'reduce-overhead')
print(f'[smoke] torch.compile mode={mode}')
forward_model = torch.compile(refiner, mode=mode)
mu_compute_mu = torch.compile(mu_model.compute_mu, mode=mode)

# ---------- 12 training steps ----------
opt = torch.optim.AdamW(refiner.parameters(), lr=1e-4)
imgs = torch.randn(4, 3, 224, 224, device=device)

step_times = []
for step in range(12):
    torch.cuda.synchronize()
    t0 = time.time()
    B = imgs.shape[0]
    N = refiner.num_patches
    noisy_mask = refiner.diffusion.generate_noisy_mask(B, N, refiner.clean_ratio, device)
    torch.compiler.cudagraph_mark_step_begin()
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        mu = mu_compute_mu(imgs, noisy_mask)
    mu = mu.clone()  # detach from mu_compute_mu's cudagraph buffer
    refiner.set_cold_context(mu, noisy_mask)
    torch.compiler.cudagraph_mark_step_begin()
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        loss, log = forward_model(imgs, epoch=0)
    opt.zero_grad()
    loss.backward()
    opt.step()
    torch.cuda.synchronize()
    dt = time.time() - t0
    step_times.append(dt)
    print(f'[smoke] step {step:2d}: loss={loss.item():.5f}  dt={dt*1000:.0f}ms')

print(f'[smoke] step times (ms): {[f"{t*1000:.0f}" for t in step_times]}')
warm = step_times[-5:]
print(f'[smoke] last-5 mean dt = {sum(warm)/len(warm)*1000:.0f}ms '
      f'(first step = {step_times[0]*1000:.0f}ms — should be 5-30x slower)')

# ---------- Checkpoint key format check ----------
sd = refiner.state_dict()
bad_keys = [k for k in sd if k.startswith('_orig_mod.') or '._orig_mod.' in k]
print(f'[smoke] state_dict has {len(sd)} keys, '
      f'_orig_mod-prefixed: {len(bad_keys)} (expect 0)')
if bad_keys:
    print(f'[smoke] FAIL: bad keys → {bad_keys[:3]}')
    sys.exit(1)

# Save then reload into a fresh, NON-compiled refiner.
ckpt_path = '/tmp/compile_smoke_ckpt.pth'
torch.save({'model': sd, 'epoch': 0}, ckpt_path)

fresh = build_subdiff_from_cfg(cfg, curriculum_cfg).to(device)
fresh_state = torch.load(ckpt_path, map_location=device, weights_only=False)['model']
if getattr(fresh.encoder, 'pos_embed_type', 'learnable') == 'sincos':
    fresh_state = {k:v for k,v in fresh_state.items() if not k.endswith('encoder.pos_embed')}
miss, unex = fresh.load_state_dict(fresh_state, strict=False)
print(f'[smoke] reload into non-compiled model: missing={len(miss)} unexpected={len(unex)}')
if len(unex) > 0:
    print(f'[smoke] FAIL: unexpected keys after reload → {unex[:3]}')
    sys.exit(1)

# Numerical match: forward through fresh model should give same loss as compiled
fresh.eval()
fresh.set_cold_context(mu, noisy_mask)
with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
    loss_fresh, _ = fresh(imgs, epoch=0)
print(f'[smoke] reload sanity: fresh forward loss = {loss_fresh.item():.5f}')

os.remove(ckpt_path)
print('[smoke] PASS')
PY
