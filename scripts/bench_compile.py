"""DDP benchmark: cold-RF refiner training step time, compile on/off.

Launched twice per PBS job (once with COMPILE=0, once with COMPILE=1) so the
two runs share the same GPUs / data / dataset cache state for a fair
comparison.

Outputs warm-step average wall time on rank 0; the difference is the real
HPC speedup from torch.compile + FA backend on the *training* hot loop.
"""

import os
import sys
import time
import yaml

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.insert(0, '/lustre1/work/c30636/test/subdiff')
from scripts.pretrain import (build_subdiff_from_cfg,
                               _enable_hpc_speedups, _print_hpc_status)


def _setup_sdpa_backend(name):
    """Pick which SDPA backend(s) to enable.
       'flash'  → PyTorch-bundled FA2 only
       'cudnn'  → cuDNN attention (FA3 on Hopper) only
       'mem'    → memory-efficient attention only
       'auto'   → all three on, runtime picks (flash wins by priority)
    """
    flags = {'flash': False, 'mem': False, 'cudnn': False}
    if name in ('flash', 'auto'):
        flags['flash'] = True
    if name in ('mem', 'auto'):
        flags['mem'] = True
    if name in ('cudnn', 'auto'):
        flags['cudnn'] = True
    torch.backends.cuda.enable_flash_sdp(flags['flash'])
    torch.backends.cuda.enable_mem_efficient_sdp(flags['mem'])
    torch.backends.cuda.enable_cudnn_sdp(flags['cudnn'])
    torch.backends.cuda.enable_math_sdp(False)
    return flags


def main():
    _enable_hpc_speedups()

    sdpa_choice = os.environ.get('SDPA', 'flash')
    sdpa_flags = _setup_sdpa_backend(sdpa_choice)

    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    dist.init_process_group('nccl')
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')
    is_main = (rank == 0)

    if is_main:
        _print_hpc_status(is_main=True)
        print(f'[BENCH] SDPA choice={sdpa_choice} → enabled={sdpa_flags}')

    torch.manual_seed(0 + rank)

    with open('configs/pretrain_vit_b8_cold_rf.yaml') as f:
        cfg = yaml.safe_load(f)
    curr = {k: cfg['curriculum'][k] for k in
            ['t_min_start','t_min_end','t_max_start','t_max_end',
             'clean_ratio_start','clean_ratio_end','warmup_epochs','schedule']}

    # Build refiner + DDP wrap (cold_rf has all params used → find_unused=False)
    refiner = build_subdiff_from_cfg(cfg, curr).to(device)
    refiner_ddp = DDP(refiner, device_ids=[local_rank],
                      find_unused_parameters=False)

    # Build frozen mu_model
    with open(cfg['model']['cold_rf_mu_config']) as f:
        mu_cfg = yaml.safe_load(f)
    mu_curr = {k: mu_cfg['curriculum'][k] for k in
               ['t_min_start','t_min_end','t_max_start','t_max_end',
                'clean_ratio_start','clean_ratio_end','warmup_epochs','schedule']}
    mu_model = build_subdiff_from_cfg(mu_cfg, mu_curr).to(device)
    mu_ckpt = torch.load(cfg['model']['cold_rf_mu_ckpt'],
                         map_location=device, weights_only=False)
    mu_state = mu_ckpt['model']
    if getattr(mu_model.encoder, 'pos_embed_type', 'learnable') == 'sincos':
        mu_state = {k: v for k, v in mu_state.items()
                    if not k.endswith('encoder.pos_embed')}
    mu_model.load_state_dict(mu_state, strict=False)
    mu_model.eval()
    for p in mu_model.parameters():
        p.requires_grad = False

    compile_enabled = os.environ.get('COMPILE', '0') == '1'
    compile_mode = os.environ.get('COMPILE_MODE', 'reduce-overhead')
    use_cudagraphs = (compile_mode == 'reduce-overhead')
    if is_main:
        print(f'\n===== rank0: compile_enabled={compile_enabled} '
              f'mode={compile_mode if compile_enabled else "n/a"} '
              f'world_size={world_size} =====')

    if compile_enabled:
        forward_model = torch.compile(refiner_ddp, mode=compile_mode)
        mu_compute_mu = torch.compile(mu_model.compute_mu, mode=compile_mode)
    else:
        forward_model = refiner_ddp
        mu_compute_mu = mu_model.compute_mu

    # Same per-GPU batch as Run 12
    B = cfg['training']['batch_size']
    opt = torch.optim.AdamW(refiner.parameters(), lr=1e-4)
    imgs = torch.randn(B, 3, 224, 224, device=device)

    NSTEPS = 40
    step_times = []
    for step in range(NSTEPS):
        torch.cuda.synchronize()
        t0 = time.time()

        N = refiner.num_patches
        noisy_mask = refiner.diffusion.generate_noisy_mask(
            imgs.shape[0], N, refiner.clean_ratio, device)
        # Cudagraph machinery is only needed under reduce-overhead mode.
        # Under default mode (no cudagraphs) the clone + mark_step_begin
        # would be pure overhead.
        if use_cudagraphs:
            torch.compiler.cudagraph_mark_step_begin()
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            mu = mu_compute_mu(imgs, noisy_mask)
        if use_cudagraphs:
            mu = mu.clone()
            torch.compiler.cudagraph_mark_step_begin()
        refiner.set_cold_context(mu, noisy_mask)

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            loss, log = forward_model(imgs, epoch=0)
        opt.zero_grad()
        loss.backward()
        opt.step()

        torch.cuda.synchronize()
        dt = time.time() - t0
        step_times.append(dt)
        if is_main and (step < 5 or step % 5 == 0):
            print(f'  step {step:2d}: loss={loss.item():.5f}  dt={dt*1000:.0f}ms')

    if is_main:
        # Skip first 10 steps (compile tracing + DDP setup + cudnn benchmark search)
        warm = step_times[10:]
        avg = sum(warm) / len(warm)
        global_imgs = B * world_size
        print(f'\n[BENCH] compile={compile_enabled}  '
              f'mode={compile_mode if compile_enabled else "n/a"}  '
              f'sdpa={sdpa_choice}  '
              f'world_size={world_size}  per_gpu_batch={B}\n'
              f'[BENCH]   first_step:           {step_times[0]*1000:7.0f}ms\n'
              f'[BENCH]   warm_avg (steps 10-39): {avg*1000:7.1f}ms\n'
              f'[BENCH]   throughput:            {global_imgs/avg:6.0f} imgs/s')

    dist.destroy_process_group()


if __name__ == '__main__':
    main()
