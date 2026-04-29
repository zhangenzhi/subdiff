"""
SubDiff pretraining script with DDP support.

Usage:
  Single GPU:
    python scripts/pretrain.py --config configs/pretrain_vit_b16.yaml

  Multi-GPU (DDP):
    torchrun --nproc_per_node=8 scripts/pretrain.py --config configs/pretrain_vit_b16.yaml
"""

import os
import sys
import argparse
import math
import yaml

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from subdiff.model import SubDiff
from subdiff.data import build_pretrain_dataloader, build_eval_dataloader
from scripts.visualize import visualize_epoch


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--resume', type=str, default=None, help='checkpoint to resume from (full state)')
    parser.add_argument('--init_from', type=str, default=None,
                        help='checkpoint to initialize weights from (partial load, '
                             'mismatched keys stay random; optimizer NOT restored)')
    return parser.parse_args()


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def build_subdiff_from_cfg(cfg, curriculum_cfg):
    return SubDiff(
        img_size=cfg['data']['image_size'],
        patch_size=cfg['model']['patch_size'],
        embed_dim=cfg['model']['embed_dim'],
        depth=cfg['model']['depth'],
        num_heads=cfg['model']['num_heads'],
        decoder_dim=cfg['model']['decoder_embed_dim'],
        decoder_depth=cfg['model']['decoder_depth'],
        decoder_num_heads=cfg['model']['decoder_num_heads'],
        num_timesteps=cfg['diffusion']['num_timesteps'],
        beta_start=cfg['diffusion']['beta_start'],
        beta_end=cfg['diffusion']['beta_end'],
        schedule_type=cfg['diffusion']['schedule_type'],
        total_epochs=cfg['training']['epochs'],
        curriculum_cfg=curriculum_cfg,
        predict_noise=cfg.get('diffusion', {}).get('predict_noise', False),
        mae_masking=cfg.get('model', {}).get('mae_masking', False),
        mask_ratio=cfg.get('model', {}).get('mask_ratio', 0.25),
        dual_decoder=cfg.get('model', {}).get('dual_decoder', False),
        clean_ratio=cfg.get('model', {}).get('clean_ratio', 0.25),
        pixel_loss_weight=cfg.get('model', {}).get('pixel_loss_weight', 1.0),
        naive_mae=cfg.get('model', {}).get('naive_mae', False),
        naive_ddpm=cfg.get('model', {}).get('naive_ddpm', False),
        qk_norm=cfg.get('model', {}).get('qk_norm', False),
        dit_minimal_head=cfg.get('model', {}).get('dit_minimal_head', False),
        use_indicators=cfg.get('model', {}).get('use_indicators', False),
        use_conv_refine=cfg.get('model', {}).get('use_conv_refine', False),
        loss_weighting=cfg.get('diffusion', {}).get('loss_weighting', 'simple'),
        snr_gamma=cfg.get('diffusion', {}).get('snr_gamma', 5.0),
        pos_embed_type=cfg.get('model', {}).get('pos_embed_type', 'sincos'),
        flow_matching=cfg.get('diffusion', {}).get('flow_matching', False),
        rf_t_sampling=cfg.get('diffusion', {}).get('rf_t_sampling', 'logit_normal'),
        rf_logit_mean=cfg.get('diffusion', {}).get('rf_logit_mean', 0.0),
        rf_logit_std=cfg.get('diffusion', {}).get('rf_logit_std', 1.0),
        rf_mae_enabled=cfg.get('diffusion', {}).get('rf_mae_enabled', False),
        rf_mae_max_mask=cfg.get('diffusion', {}).get('rf_mae_max_mask', 0.5),
        mae_aux_weight=cfg.get('diffusion', {}).get('mae_aux_weight', 0.1),
        cold_rf=cfg.get('model', {}).get('cold_rf', False),
    )


def cosine_lr_schedule(optimizer, epoch, total_epochs, warmup_epochs, lr, min_lr,
                       schedule='cosine'):
    """Returns current lr. Supports 'cosine' (default, with warmup + cosine decay)
    and 'constant' (always returns lr, ignores warmup / min_lr — DDPM/DiT standard)."""
    if schedule == 'constant':
        cur_lr = lr
    else:
        if epoch < warmup_epochs:
            cur_lr = lr * (epoch + 1) / max(warmup_epochs, 1)
        else:
            progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
            cur_lr = min_lr + (lr - min_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))
    for param_group in optimizer.param_groups:
        param_group['lr'] = cur_lr
    return cur_lr


def setup_distributed():
    if 'RANK' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        dist.init_process_group('nccl')
        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank, True
    return 0, 1, 0, False


def _enable_hpc_speedups():
    """Lightweight HPC perf flags. Safe defaults, no algorithmic changes."""
    if not torch.cuda.is_available():
        return
    # TF32 matmul on Ampere/Hopper (1.3-1.5x on fp32 ops not covered by autocast)
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    # Let cuDNN pick the fastest conv algorithm (matters for patch_embed)
    torch.backends.cudnn.benchmark = True
    # Prefer Flash Attention backend for SDPA on H100. PyTorch 2.4+ on Hopper
    # auto-selects FA3 over FA2 when flash_sdp is enabled and dtype is bf16/fp16.
    try:
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(False)
    except AttributeError:
        pass  # older torch


def _select_sdpa_backend(name):
    """Force a specific SDPA backend. Default behavior (without calling this)
    is whatever PyTorch picks — on H100 + bf16 that's FA2.

    name: 'flash' | 'cudnn' | 'mem' | 'auto'
      - 'flash':  PyTorch-bundled FlashAttention-2 only
      - 'cudnn':  cuDNN attention (= FlashAttention-3 on Hopper sm_90)
      - 'mem':    memory-efficient attention only
      - 'auto':   all three on; runtime picks (flash wins by priority)

    Bench (cold-RF, B=64/GPU, 2× H100):
      compile=default + flash → 150.3ms/step (1.86× vs eager+flash)
      compile=default + cudnn → 139.5ms/step (2.01× vs eager+flash, 1.08× extra)
    """
    if name is None:
        return  # leave whatever _enable_hpc_speedups set
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


def _print_hpc_status(is_main):
    """One-shot dump of HPC-relevant runtime state. Helps confirm FA3 path
    is reachable (Hopper sm_90 + flash_sdp=True + bf16 → FA3 used at runtime)."""
    if not is_main or not torch.cuda.is_available():
        return
    cap = torch.cuda.get_device_capability(0)
    print(f"[HPC] PyTorch {torch.__version__}, "
          f"GPU={torch.cuda.get_device_name(0)} (sm_{cap[0]}{cap[1]})")
    flags = {
        'flash': torch.backends.cuda.flash_sdp_enabled(),
        'mem_efficient': torch.backends.cuda.mem_efficient_sdp_enabled(),
        'math': torch.backends.cuda.math_sdp_enabled(),
    }
    try:
        flags['cudnn'] = torch.backends.cuda.cudnn_sdp_enabled()
    except AttributeError:
        pass
    print(f"[HPC] SDPA backends enabled: {flags}")
    print(f"[HPC] TF32 matmul={torch.backends.cuda.matmul.allow_tf32}, "
          f"cudnn TF32={torch.backends.cudnn.allow_tf32}, "
          f"cudnn benchmark={torch.backends.cudnn.benchmark}")


def main():
    args = get_args()
    cfg = load_config(args.config)

    _enable_hpc_speedups()
    sdpa_backend = cfg.get('training', {}).get('sdpa_backend', None)
    _select_sdpa_backend(sdpa_backend)

    rank, world_size, local_rank, distributed = setup_distributed()
    is_main = (rank == 0)
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')

    _print_hpc_status(is_main)
    if is_main and sdpa_backend is not None:
        print(f"[HPC] sdpa_backend override: {sdpa_backend}")

    # Build model
    curriculum_cfg = {
        't_min_start': cfg['curriculum']['t_min_start'],
        't_min_end': cfg['curriculum']['t_min_end'],
        't_max_start': cfg['curriculum']['t_max_start'],
        't_max_end': cfg['curriculum']['t_max_end'],
        'clean_ratio_start': cfg['curriculum']['clean_ratio_start'],
        'clean_ratio_end': cfg['curriculum']['clean_ratio_end'],
        'warmup_epochs': cfg['curriculum']['warmup_epochs'],
        'schedule': cfg['curriculum']['schedule'],
    }

    model = build_subdiff_from_cfg(cfg, curriculum_cfg).to(device)

    # Cold-RF (Run 12): build a SEPARATE frozen mu_model from a different
    # config + ckpt. Its only role is to provide μ via compute_mu(); we
    # don't register it as a submodule on the trainable model.
    mu_model = None
    if cfg.get('model', {}).get('cold_rf', False):
        mu_cfg_path = cfg['model']['cold_rf_mu_config']
        mu_ckpt_path = cfg['model']['cold_rf_mu_ckpt']
        with open(mu_cfg_path) as f:
            mu_cfg = yaml.safe_load(f)
        mu_curriculum_cfg = {
            't_min_start': mu_cfg['curriculum']['t_min_start'],
            't_min_end': mu_cfg['curriculum']['t_min_end'],
            't_max_start': mu_cfg['curriculum']['t_max_start'],
            't_max_end': mu_cfg['curriculum']['t_max_end'],
            'clean_ratio_start': mu_cfg['curriculum']['clean_ratio_start'],
            'clean_ratio_end': mu_cfg['curriculum']['clean_ratio_end'],
            'warmup_epochs': mu_cfg['curriculum']['warmup_epochs'],
            'schedule': mu_cfg['curriculum']['schedule'],
        }
        mu_model = build_subdiff_from_cfg(mu_cfg, mu_curriculum_cfg).to(device)
        mu_ckpt = torch.load(mu_ckpt_path, map_location=device, weights_only=False)
        state = mu_ckpt['model']
        if getattr(mu_model.encoder, 'pos_embed_type', 'learnable') == 'sincos':
            state = {k: v for k, v in state.items()
                     if not k.endswith('encoder.pos_embed')}
        mu_model.load_state_dict(state, strict=False)
        mu_model.eval()
        for p in mu_model.parameters():
            p.requires_grad = False
        if is_main:
            print(f"Loaded frozen mu_model from {mu_ckpt_path} "
                  f"(ep {mu_ckpt.get('epoch', '?')}, "
                  f"avg_loss={mu_ckpt.get('avg_loss', 'n/a')})")

    if is_main:
        param_count = sum(p.numel() for p in model.parameters()) / 1e6
        if getattr(model, 'cold_rf', False):
            mode = "cold-RF refiner (v-head only, μ from frozen mu_model)"
        elif model.naive_ddpm:
            head_type = "minimal head" if model.dit_minimal_head else "4-layer decoder"
            if model.flow_matching:
                if model.rf_mae_enabled:
                    obj = (f"RF v-pred + MAE mask "
                           f"(r≤{model.rf_mae_max_mask}, aux={model.mae_aux_weight})")
                else:
                    obj = "RF v-pred"
            else:
                obj = "DDPM eps-pred"
            mode = f"naive-ViT ({obj}, {head_type})"
        elif model.naive_mae:
            mode = f"naive MAE (mask={model.mask_ratio})"
        elif model.dual_decoder:
            mode = f"dual-decoder (clean={model.clean_ratio}, eps + {model.pixel_loss_weight}*pix)"
        else:
            tgt = "noise" if model.predict_noise else "pixel"
            mode = f"MAE-masked ({model.mask_ratio}) + {tgt}" if model.mae_masking else tgt
        qk = " + QK-Norm" if cfg.get('model', {}).get('qk_norm', False) else ""
        print(f"SubDiff model: {param_count:.1f}M parameters, mode: {mode}{qk}")
        print(f"Curriculum: {model.curriculum}")

    if distributed:
        # find_unused_parameters=True for MAE-masking / dual-decoder modes
        # where some decoder params may not contribute to loss each step.
        # Cold-RF (Run 12) Refiner has v-head only — all params used; setting
        # find_unused=False here is required for torch.compile to avoid graph
        # breaks from the unused-param scan.
        find_unused = not cfg.get('model', {}).get('cold_rf', False)
        if is_main:
            print(f"DDP find_unused_parameters={find_unused}")
        model = DDP(model, device_ids=[local_rank],
                    find_unused_parameters=find_unused)

    model_without_ddp = model.module if distributed else model
    model_raw = model_without_ddp

    # ----------------------------------------------------------------------
    # torch.compile (opt-in via training.compile in config). Numerics-equivalent
    # HPC speedup. CRITICAL: state_dict / optimizer / DDP all keep using `model`
    # and `model_raw`, which point to the un-compiled SubDiff. The compiled
    # wrapper `forward_model` is used ONLY for the forward call. This way:
    #   - `model_raw.state_dict()` saves clean keys (no _orig_mod. prefix)
    #   - existing checkpoints load without key-rewriting
    #   - turning compile on/off across runs is transparent
    # ----------------------------------------------------------------------
    compile_enabled = cfg.get('training', {}).get('compile', False)
    compile_mode = cfg.get('training', {}).get('compile_mode', 'default')
    # Only reduce-overhead uses CUDA graphs and therefore needs the
    # mark_step_begin / clone machinery around the cross-graph mu handoff.
    # Under 'default' those would be pure overhead.
    use_cudagraphs = compile_enabled and compile_mode == 'reduce-overhead'
    if compile_enabled:
        if is_main:
            print(f"[torch.compile] enabled mode={compile_mode}  "
                  f"cudagraphs={use_cudagraphs}. "
                  f"First step will be slow (graph tracing).")
        forward_model = torch.compile(model, mode=compile_mode)
        if mu_model is not None:
            mu_compute_mu = torch.compile(mu_model.compute_mu, mode=compile_mode)
        else:
            mu_compute_mu = None
    else:
        forward_model = model
        mu_compute_mu = mu_model.compute_mu if mu_model is not None else None

    # Build dataloaders
    backend = cfg.get('data', {}).get('backend', 'torch')
    # Auto-pick transform: generation modes use diffusion-style augmentation
    # (no aggressive RandomResizedCrop). Explicit config override takes priority.
    default_tt = 'diffusion' if cfg.get('model', {}).get('naive_ddpm', False) else 'ssl'
    transform_type = cfg.get('data', {}).get('transform_type', default_tt)
    if is_main:
        print(f"Dataloader: backend={backend}, transform_type={transform_type}, "
              f"num_workers={cfg['data']['num_workers']}")
    train_loader, train_sampler = build_pretrain_dataloader(
        imagenet_dir=cfg['data']['imagenet_dir'],
        image_size=cfg['data']['image_size'],
        batch_size=cfg['training']['batch_size'],
        num_workers=cfg['data']['num_workers'],
        distributed=distributed,
        backend=backend,
        transform_type=transform_type,
    )

    # Val images for periodic visualization (always use torch backend; one-shot)
    vis_imgs = None
    if is_main:
        val_loader, _ = build_eval_dataloader(
            imagenet_dir=cfg['data']['imagenet_dir'],
            image_size=cfg['data']['image_size'],
            batch_size=8, num_workers=2,
            backend='torch',
        )
        vis_imgs, _ = next(iter(val_loader))

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg['training']['lr'],
        weight_decay=cfg['training']['weight_decay'],
        betas=(0.9, 0.95),
    )

    # Tensorboard
    writer = None
    if is_main:
        os.makedirs(cfg['logging']['log_dir'], exist_ok=True)
        writer = SummaryWriter(cfg['logging']['log_dir'])

    # Mixed precision (bf16 on H100)
    use_amp = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    amp_dtype = torch.bfloat16 if use_amp else torch.float32
    scaler = torch.amp.GradScaler('cuda', enabled=(use_amp and amp_dtype == torch.float16))
    # bf16 doesn't need GradScaler, but we keep it disabled for clean code path
    if is_main:
        print(f"Mixed precision: {amp_dtype}, GradScaler: {scaler.is_enabled()}")

    # Resume (full state restore) OR init_from (partial weight load)
    start_epoch = 0
    best_loss = float('inf')
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        # Skip loading pos_embed when using fixed sin-cos (requires_grad=False):
        # otherwise the checkpoint's (learned) pos_embed would overwrite our
        # fresh sin-cos values.
        state_to_load = ckpt['model']
        if getattr(model_raw.encoder, 'pos_embed_type', 'learnable') == 'sincos':
            state_to_load = {k: v for k, v in state_to_load.items()
                             if not k.endswith('encoder.pos_embed')}
        model_raw.load_state_dict(state_to_load, strict=False)
        # Only restore optimizer if param count matches (architecture unchanged)
        try:
            optimizer.load_state_dict(ckpt['optimizer'])
        except (ValueError, KeyError):
            if is_main:
                print("WARNING: optimizer state not restored (param group mismatch)")
        if 'scaler' in ckpt and scaler.is_enabled():
            scaler.load_state_dict(ckpt['scaler'])
        start_epoch = ckpt['epoch'] + 1
        best_loss = ckpt.get('best_loss', float('inf'))
        if is_main:
            print(f"Resumed from epoch {start_epoch}, best_loss={best_loss:.4f}")
    elif args.init_from:
        ckpt = torch.load(args.init_from, map_location=device)
        src_state = ckpt['model']
        tgt_state = model_raw.state_dict()
        loaded, skipped_shape, skipped_missing = [], [], []
        for k, v in tgt_state.items():
            if k in src_state and src_state[k].shape == v.shape:
                tgt_state[k] = src_state[k]
                loaded.append(k)
            elif k in src_state:
                skipped_shape.append(k)
            else:
                skipped_missing.append(k)
        model_raw.load_state_dict(tgt_state)
        if is_main:
            print(f"Initialized from {args.init_from} (epoch {ckpt.get('epoch', '?')})")
            print(f"  Loaded {len(loaded)} tensors")
            if skipped_shape:
                print(f"  Skipped (shape mismatch): {len(skipped_shape)} — "
                      f"e.g., {skipped_shape[:3]}")
            if skipped_missing:
                print(f"  Random init (not in source): {len(skipped_missing)} — "
                      f"e.g., {skipped_missing[:3]}")

    # Training loop
    import time
    total_epochs = cfg['training']['epochs']
    print_every = cfg['logging']['print_every']
    save_every = cfg['logging']['save_every']
    batch_size_per_gpu = cfg['training']['batch_size']
    effective_bs = batch_size_per_gpu * world_size

    for epoch in range(start_epoch, total_epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        cur_lr = cosine_lr_schedule(
            optimizer, epoch, total_epochs,
            cfg['training']['warmup_epochs'],
            cfg['training']['lr'],
            cfg['training']['min_lr'],
            schedule=cfg['training'].get('lr_schedule', 'cosine'),
        )

        # Get curriculum state for logging
        curriculum_state = model_raw.curriculum.get_state(epoch)

        model.train()
        total_loss = 0.0
        num_steps = 0

        # Per-step timing breakdown (data IO vs compute)
        t_data_total = 0.0
        t_step_total = 0.0
        epoch_start = time.time()
        last_log_time = time.time()
        last_log_step = 0
        t_iter_start = time.time()

        for step, (imgs, _) in enumerate(train_loader):
            t_data = time.time() - t_iter_start  # time spent waiting for data
            imgs = imgs.to(device, non_blocking=True)

            # Cold-RF (Run 12): pre-compute μ from frozen mu_model and the
            # per-step noisy mask, then attach to the trainable refiner so
            # _forward_cold_rf can read them. This keeps the DDP forward
            # signature intact (still just forward(imgs, epoch)).
            #
            # cudagraphs path only: mu_compute_mu's output lives in a
            # cudagraph-managed buffer that can be recycled before refiner
            # reads it. Insert mark_step_begin + clone to break the alias.
            # mode='default' has no cudagraphs → these would be pure overhead
            # (and bench showed they're costly enough to flip 1.86× speedup
            # into 0.46× regression).
            if mu_model is not None:
                B_ = imgs.shape[0]
                N_ = model_raw.num_patches
                noisy_mask_ = model_raw.diffusion.generate_noisy_mask(
                    B_, N_, model_raw.clean_ratio, device)
                if use_cudagraphs:
                    torch.compiler.cudagraph_mark_step_begin()
                with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=use_amp):
                    mu_ = mu_compute_mu(imgs, noisy_mask_)
                if use_cudagraphs:
                    mu_ = mu_.clone()
                model_raw.set_cold_context(mu_, noisy_mask_)
                if use_cudagraphs:
                    torch.compiler.cudagraph_mark_step_begin()

            with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=use_amp):
                if compile_enabled:
                    # Direct kwargs call — modern DDP supports kwargs natively,
                    # and avoids _forward_ddp's patched_forward hack which would
                    # trigger torch.compile to recompile every step.
                    loss, log_dict = forward_model(imgs, epoch=epoch)
                elif distributed:
                    loss, log_dict = _forward_ddp(model, imgs, epoch)
                else:
                    loss, log_dict = model(imgs, epoch=epoch)

            optimizer.zero_grad()
            scaler.scale(loss).backward()

            if cfg['training']['clip_grad'] > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), cfg['training']['clip_grad'])

            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            num_steps += 1
            global_step = epoch * len(train_loader) + step
            t_step = time.time() - t_iter_start  # full step time
            t_data_total += t_data
            t_step_total += t_step

            if is_main and step % print_every == 0:
                # throughput stats since last log
                now = time.time()
                steps_since = max(step - last_log_step, 1)
                wall = now - last_log_time
                steps_per_sec = steps_since / max(wall, 1e-6)
                imgs_per_sec = steps_per_sec * effective_bs
                util = 1.0 - (t_data / max(t_step, 1e-6))  # GPU util proxy
                print(f"Epoch [{epoch}/{total_epochs}] Step [{step}/{len(train_loader)}] "
                      f"loss={log_dict['loss'].item():.4f} "
                      f"noisy_loss={log_dict['noisy_loss'].item():.4f} "
                      f"clean_loss={log_dict['clean_loss'].item():.4f} "
                      f"t=[{curriculum_state['t_min']},{curriculum_state['t_max']}] "
                      f"clean_ratio={curriculum_state['clean_ratio']:.3f} "
                      f"lr={cur_lr:.6f} "
                      f"| {steps_per_sec:.2f} step/s {imgs_per_sec:.0f} img/s "
                      f"data={t_data*1000:.0f}ms step={t_step*1000:.0f}ms util={util:.0%}")
                last_log_time = now
                last_log_step = step

            if writer and step % print_every == 0:
                writer.add_scalar('train/loss', log_dict['loss'].item(), global_step)
                writer.add_scalar('train/noisy_loss', log_dict['noisy_loss'].item(), global_step)
                writer.add_scalar('train/clean_loss', log_dict['clean_loss'].item(), global_step)
                writer.add_scalar('train/lr', cur_lr, global_step)
                writer.add_scalar('curriculum/t_min', curriculum_state['t_min'], global_step)
                writer.add_scalar('curriculum/t_max', curriculum_state['t_max'], global_step)
                writer.add_scalar('curriculum/clean_ratio', curriculum_state['clean_ratio'], global_step)
                writer.add_scalar('curriculum/t_mean', log_dict['t_mean'].item(), global_step)
                writer.add_scalar('perf/step_ms', t_step * 1000, global_step)
                writer.add_scalar('perf/data_ms', t_data * 1000, global_step)

            t_iter_start = time.time()

        avg_loss = total_loss / max(num_steps, 1)
        epoch_wall = time.time() - epoch_start
        avg_data_pct = (t_data_total / max(t_step_total, 1e-6)) * 100
        avg_step_ms = t_step_total / max(num_steps, 1) * 1000
        epoch_imgs_per_sec = (num_steps * effective_bs) / max(epoch_wall, 1e-6)
        if is_main:
            print(f"Epoch [{epoch}/{total_epochs}] avg_loss={avg_loss:.4f} "
                  f"| wall={epoch_wall:.0f}s ({epoch_wall/60:.1f}min) "
                  f"avg_step={avg_step_ms:.0f}ms "
                  f"throughput={epoch_imgs_per_sec:.0f} img/s "
                  f"data_overhead={avg_data_pct:.1f}%")
            if writer:
                writer.add_scalar('perf/epoch_wall_sec', epoch_wall, epoch)
                writer.add_scalar('perf/epoch_imgs_per_sec', epoch_imgs_per_sec, epoch)
                writer.add_scalar('perf/data_overhead_pct', avg_data_pct, epoch)

        # Save checkpoints: latest (every save_every) + best (when avg_loss improves)
        if is_main and (epoch + 1) % save_every == 0:
            ckpt_dir = os.path.join(cfg['logging']['log_dir'], 'checkpoints')
            os.makedirs(ckpt_dir, exist_ok=True)

            is_best = avg_loss < best_loss
            if is_best:
                best_loss = avg_loss

            state = {
                'model': model_raw.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scaler': scaler.state_dict(),
                'epoch': epoch,
                'avg_loss': avg_loss,
                'best_loss': best_loss,
                'config': cfg,
            }
            latest_path = os.path.join(ckpt_dir, 'checkpoint_latest.pth')
            torch.save(state, latest_path)
            print(f"Saved latest: {latest_path} (avg_loss={avg_loss:.4f})")

            if is_best:
                best_path = os.path.join(ckpt_dir, 'checkpoint_best.pth')
                torch.save(state, best_path)
                print(f"Saved best: {best_path} (avg_loss={avg_loss:.4f})")

            # Visualize at this epoch
            vis_dir = os.path.join(cfg['logging']['log_dir'], 'visualizations')
            visualize_epoch(model_raw, vis_imgs, epoch, vis_dir, device, n_samples=4)

    # Save final as latest (best is kept separately)
    if is_main:
        ckpt_dir = os.path.join(cfg['logging']['log_dir'], 'checkpoints')
        os.makedirs(ckpt_dir, exist_ok=True)
        latest_path = os.path.join(ckpt_dir, 'checkpoint_latest.pth')
        torch.save({
            'model': model_raw.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scaler': scaler.state_dict(),
            'epoch': total_epochs - 1,
            'avg_loss': avg_loss,
            'best_loss': best_loss,
            'config': cfg,
        }, latest_path)
        print(f"Saved final as latest: {latest_path} (best_loss={best_loss:.4f})")
        vis_dir = os.path.join(cfg['logging']['log_dir'], 'visualizations')
        visualize_epoch(model_raw, vis_imgs, total_epochs - 1, vis_dir, device, n_samples=4)

    if distributed:
        dist.destroy_process_group()


def _forward_ddp(model, imgs, epoch):
    """Forward pass through DDP model.

    We override the forward to pass epoch through. The DDP wrapper calls
    model.module.forward, so we temporarily store epoch.
    """
    model.module._current_epoch = epoch
    # Use a simple wrapper
    loss, log_dict = _SubDiffDDPForward.apply(model, imgs, epoch)
    return loss, log_dict


class _SubDiffDDPForward:
    """Helper to run forward through DDP for gradient sync."""

    @staticmethod
    def apply(model, imgs, epoch):
        # DDP needs the forward to go through the wrapper
        # We use a hook approach: the model.forward calls module.forward
        # Store epoch so module can access it
        model.module._ddp_epoch = epoch
        # Patch forward temporarily
        original_forward = model.module.forward

        def patched_forward(x):
            return original_forward(x, epoch=model.module._ddp_epoch)

        model.module.forward = patched_forward
        loss, log_dict = model(imgs)
        model.module.forward = original_forward
        return loss, log_dict


if __name__ == '__main__':
    main()
